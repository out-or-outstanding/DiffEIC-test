import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Mapping, Any, List
import pyiqa

import os
import math
import einops
import numpy as np
from PIL import Image

from transformers import CLIPTextModel, CLIPTokenizer
from .diffeic import DiffEIC, instantiate_from_config
from .spaced_sampler import SpacedSampler
from ldm.util import log_txt_as_img, exists, default
from utils.utils import write_body, read_body, img2tensor, Path

# ---------------------- 全局显存优化配置 ----------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.cuda.empty_cache()

# ---------------------- 兼容低版本PyTorch的梯度检查点装饰器（修复版） ----------------------
def checkpoint_module(cls):
    original_forward = cls.forward
    
    def checkpoint_forward(self, *args, **kwargs):
        # 只在训练模式下启用checkpoint
        if not self.training:
            return original_forward(self, *args, **kwargs)
        
        # 检查是否有梯度需求
        has_grad = False
        all_inputs = args + tuple(kwargs.values())
        for inp in all_inputs:
            if isinstance(inp, torch.Tensor) and inp.requires_grad:
                has_grad = True
                break
        
        if not has_grad:
            return original_forward(self, *args, **kwargs)
        
        # 修复：PyTorch 2.0.1兼容的嵌套checkpoint检测
        # 使用更简单的方法：直接移除嵌套checkpoint，只保留外层checkpoint
        try:
            # 使用重入式checkpoint（PyTorch 2.0.1默认）
            return torch.utils.checkpoint.checkpoint(
                original_forward, self, *args, **kwargs, 
                preserve_rng_state=False
            )
        except (AttributeError, RecursionError, TypeError, RuntimeError):
            # 如果checkpoint失败，回退到原始forward
            return original_forward(self, *args, **kwargs)
    
    cls.forward = checkpoint_forward
    return cls

class DynamicBackgroundCorrection(nn.Module):
    def __init__(self, in_channels, delta_channels=8):
        super().__init__()
        self.in_channels = in_channels
        self.delta_channels = delta_channels
        self.encoder_delta = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, 1, padding=0),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels//2, delta_channels*2, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(delta_channels*2, delta_channels, 3, padding=1, stride=2),
            nn.LeakyReLU(inplace=True)
        )
        self.decoder_delta = nn.Sequential(
            # 显式指定输出尺寸，避免stride=2导致的尺寸计算错误
            nn.ConvTranspose2d(delta_channels, delta_channels*2, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(delta_channels*2, in_channels, 3, padding=1),
            nn.Tanh()
        )

    def encode(self, z_bg_prior, z_bg_curr):
        delta_raw = z_bg_curr - z_bg_prior
        delta_raw = (delta_raw - delta_raw.mean(dim=[2,3], keepdim=True)) / (delta_raw.std(dim=[2,3], keepdim=True) + 1e-6)
        delta_feat = self.encoder_delta(delta_raw)
        # 断言：确保delta_feat尺寸正确（32x32）
        assert delta_feat.shape[2:] == (32, 32), f"delta_feat尺寸错误：{delta_feat.shape[2:]}"
        del delta_raw
        return delta_feat

    def decode(self, delta_quant, z_bg_prior):
        delta_raw_rec = self.decoder_delta(delta_quant)
        # 断言：确保解码后尺寸与z_bg_prior一致（64x64）
        assert delta_raw_rec.shape == z_bg_prior.shape, f"解码尺寸不匹配：{delta_raw_rec.shape} vs {z_bg_prior.shape}"
        delta_raw_rec = torch.clamp(delta_raw_rec, -0.5, 0.5)
        z_bg_corrected = z_bg_prior + delta_raw_rec
        del delta_raw_rec
        return z_bg_corrected

class BiDirectionalCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_heads = self._get_valid_heads(num_heads, dim)
        self.head_dim = dim // self.num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def _get_valid_heads(self, num_heads, dim):
        max_heads = min(num_heads, dim)
        if dim < max_heads:
            return dim
        for i in range(max_heads, 0, -1):
            if dim % i == 0:
                return i
        return 1

    def forward(self, q, kv):
        B, C_q, H, W = q.shape
        B, C_kv, H_kv, W_kv = kv.shape
        
        # 保存原始q的尺寸用于断言
        original_q_shape = q.shape
        
        # 确保空间尺寸一致（强制插值到q的尺寸）
        if (H_kv, W_kv) != (H, W):
            kv = F.interpolate(kv, size=(H, W), mode='bilinear', align_corners=False)
            assert kv.shape[2:] == (H, W), f"插值后尺寸错误：{kv.shape[2:]} vs {(H, W)}"
        
        # 通道数对齐（预先创建层，避免动态创建）
        if C_q != self.dim:
            q = F.adaptive_avg_pool2d(q, (self.dim, 1)) if C_q > self.dim else F.pad(q, (0,0,0,0,0,self.dim-C_q))
        if C_kv != self.dim:
            kv = F.adaptive_avg_pool2d(kv, (self.dim, 1)) if C_kv > self.dim else F.pad(kv, (0,0,0,0,0,self.dim-C_kv))
        
        # 展平序列 - 添加contiguous确保内存布局
        q_flat = einops.rearrange(q, 'b c h w -> b (h w) c').contiguous()
        kv_flat = einops.rearrange(kv, 'b c h w -> b (h w) c').contiguous()
        
        # 投影
        q = self.q_proj(q_flat)
        k = self.k_proj(kv_flat)
        v = self.v_proj(kv_flat)
        del q_flat, kv_flat
        
        # 多头拆分 - 添加contiguous确保内存布局
        q = einops.rearrange(q, 'b s (h d) -> b h s d', h=self.num_heads, d=self.head_dim).contiguous()
        k = einops.rearrange(k, 'b s (h d) -> b h s d', h=self.num_heads, d=self.head_dim).contiguous()
        v = einops.rearrange(v, 'b s (h d) -> b h s d', h=self.num_heads, d=self.head_dim).contiguous()
        
        # 注意力计算
        attn_output = F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=False
        )
        
        del q, k, v
        
        # 还原形状 - 添加contiguous确保内存布局
        attn_output = einops.rearrange(attn_output, 'b h s d -> b s (h d)').contiguous()
        out = self.out_proj(attn_output)
        del attn_output
        
        out = einops.rearrange(out, 'b (h w) c -> b c h w', h=H, w=W).contiguous()
        # 修复：使用保存的原始q尺寸进行断言
        assert out.shape == original_q_shape, f"注意力输出尺寸错误：{out.shape} vs {original_q_shape}"
        return out

# 修复InnovativeCrossModalFusion中的内存布局问题
class InnovativeCrossModalFusion(nn.Module):
    def __init__(self, img_dim=4, txt_dim=512, num_heads=4):
        super().__init__()
        self.img_dim = img_dim
        self.txt_dim = txt_dim
        
        # 修复：预先创建所有对齐层，避免动态创建
        self.img_align = nn.Conv2d(4, img_dim, 1, bias=False)  # x_noisy默认4通道
        self.bg_align = nn.Conv2d(128, img_dim, 1, bias=False) # bg_encoder输出128通道
        self.txt_proj = nn.Sequential(
            nn.Linear(self.txt_dim, img_dim, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Unflatten(1, (img_dim, 1, 1))
        )
        
        self.attn_img2bg_txt = BiDirectionalCrossAttention(img_dim, num_heads)
        self.attn_bg_txt2img = BiDirectionalCrossAttention(img_dim, num_heads)
        self.channel_reduce = nn.Conv2d(img_dim * 2, img_dim, 1, bias=False)
        self.bg_channel_align = nn.Conv2d(img_dim, img_dim, 1, bias=False)
        self.txt_channel_align = nn.Conv2d(img_dim, img_dim, 1, bias=False)
        self.gating = ModalityAwareGating(img_dim)
        self.res_calibrate = nn.Conv2d(img_dim, img_dim, 1, bias=False)
        self.output_align = nn.Conv2d(img_dim, img_dim, 1, bias=False)

    def forward(self, z_img, z_bg, z_txt):
        B, C_img, H_img, W_img = z_img.shape
        B, C_bg, H_bg, W_bg = z_bg.shape
        
        # 1. 固定通道对齐（不再动态创建层）
        z_img = self.img_align(z_img)  # 强制转为img_dim通道
        z_bg = self.bg_align(z_bg)      # 强制转为img_dim通道
        
        # --- 新增：将背景特征对齐到当前latent（z_img）的空间分辨率 ---
        # 这解决了训练时256x256输入导致的32x32 vs 64x64尺寸不匹配问题
        if (H_bg, W_bg) != (H_img, W_img):
            z_bg = F.interpolate(
                z_bg, size=(H_img, W_img), mode='bilinear', align_corners=False
            ).contiguous()  # 添加contiguous确保内存布局
        
        # 2. 文本特征处理（现在直接repeat到z_img的目标尺寸）
        z_txt_align = self.txt_proj(z_txt)                    # (B, img_dim, 1, 1)
        z_txt_align = z_txt_align.repeat(1, 1, H_img, W_img).contiguous()  # 添加contiguous
        
        # 3. 拼接+降维（kv现在与z_img空间尺寸一致）
        kv = torch.cat([z_bg, z_txt_align], dim=1).contiguous()  # 添加contiguous
        kv = self.channel_reduce(kv)
        
        # 4. 正向注意力：img → (bg + txt)
        attn_img = self.attn_img2bg_txt(z_img, kv)  # 输出尺寸必然匹配z_img
        
        # 5. 反向注意力：img → (bg + txt) （query仍是z_img，因此输出仍匹配z_img尺寸）
        q_bg_txt = torch.cat([z_bg, z_txt_align], dim=1).contiguous()  # 添加contiguous
        q_bg_txt = self.channel_reduce(q_bg_txt)
        attn_bg_txt = self.attn_bg_txt2img(z_img, q_bg_txt)

        # 原代码逻辑：这里将反向注意力输出同时用作attn_bg和attn_txt
        # （可能设计意图是共享反向注意力特征，若需分离可进一步拆分）
        attn_bg = attn_bg_txt
        attn_txt = attn_bg_txt
        
        # 通道对齐（输入已是img_dim，无需额外操作，但保留层以保持可扩展性）
        z_bg_aligned = self.bg_channel_align(z_bg)
        z_txt_aligned = self.txt_channel_align(z_txt_align)
        
        # 7. 门控融合（所有张量现在空间尺寸一致：H_img x W_img）
        weights = self.gating(z_img)
        # gating输出形状为 (B, 3)，不需要unsqueeze
        α, β, γ = weights.chunk(3, dim=-1)                    # 每个 (B, 1)
        α = α.view(B, 1, 1, 1).contiguous()  # 添加contiguous
        β = β.view(B, 1, 1, 1).contiguous()  # 添加contiguous
        γ = γ.view(B, 1, 1, 1).contiguous()  # 添加contiguous
        
        z_fused = α * attn_img + β * (z_bg_aligned + attn_bg) + γ * (z_txt_aligned + attn_txt)
        
        # 8. 残差连接与输出对齐
        z_fused = self.res_calibrate(z_fused) + z_img
        z_fused = self.output_align(z_fused)
        
        # 清理中间变量（可选，节省显存）
        del (α, β, γ, weights, attn_img, attn_bg, attn_txt,
                z_txt_align, kv, q_bg_txt, z_bg_aligned, z_txt_aligned)
        
        return z_fused

class ModalityAwareGating(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        hidden_dim = max(dim // 8, 1)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=-1)
        )

    def forward(self, z_img):
        B, C = z_img.shape[:2]
        feat_gap = self.gap(z_img).view(B, C)
        out = self.mlp(feat_gap)
        del feat_gap
        return out

class MonitorBGEncoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=128, freeze=True):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, out_channels, 3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        
        self.fusion = nn.Conv2d(out_channels + 64, out_channels, 1, bias=False)
        self.disturb_conv = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=2, dilation=2, bias=False)
        
        if freeze:
            for param in self.parameters():
                param.requires_grad = False
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m == self.disturb_conv:
                    nn.init.normal_(m.weight, mean=0, std=0.01)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # 断言：输入尺寸正确（512x512）
        assert x.shape[2:] in [(256, 256), (512, 512)], f"BG编码器输入尺寸错误：{x.shape[2:]}"
        
        feat1 = self.conv1(x)  # 512→256
        feat2 = self.conv2(feat1)  # 256→128
        feat3 = self.conv3(feat2)  # 128→128
        
        feat_fused = self.fusion(torch.cat([feat3, feat2], dim=1))
        z_bg = self.disturb_conv(feat_fused) + feat3
        
        # 强制降采样到64x64（避免尺寸不一致）
        if z_bg.shape[2] > 64:
            z_bg = F.avg_pool2d(z_bg, kernel_size=2, stride=2)
        
        # 断言：输出尺寸正确（64x64）
        assert z_bg.shape[2:] == (64, 64), f"BG编码器输出尺寸错误：{z_bg.shape[2:]}"
        
        del feat1, feat2, feat3, feat_fused
        return z_bg

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained("weight/clip_vit_base_patch16")
        self.clip_txt = CLIPTextModel.from_pretrained("weight/clip_vit_base_patch16")
        for param in self.clip_txt.parameters():
            param.requires_grad = False

    def forward(self, text_list):
        tokens = self.tokenizer(
            text_list, return_tensors="pt", padding=True, truncation=True
        ).to(next(self.parameters()).device)
        out = self.clip_txt(**tokens).pooler_output
        # 断言：文本特征维度正确（512）
        assert out.shape[-1] == 512, f"文本特征维度错误：{out.shape[-1]}"
        return out

class DiffEICWithPriorAndText(DiffEIC):
    def __init__(
        self, 
        control_stage_config: Mapping[str, Any], 
        control_key: str,
        sd_locked: bool,
        learning_rate: float,
        aux_learning_rate: float,
        l_bpp_weight: float,
        l_guide_weight: float,
        sync_path: str, 
        synch_control: bool,
        ckpt_path_pre: str,
        preprocess_config: Mapping[str, Any],
        calculate_metrics: Mapping[str, Any],
        bg_encoder_out_channels: int = 128,
        dbc_delta_channels: int = 8,
        cmf_img_dim: int = 4,
        cmf_txt_dim: int = 512,
        cmf_num_heads: int = 4,
        first_stage_key: str = 'jpg',
        *args, 
        **kwargs
    ) -> "DiffEICWithPriorAndText":
        super().__init__(
            control_stage_config=control_stage_config,
            control_key=control_key,
            sd_locked=sd_locked,
            learning_rate=learning_rate,
            aux_learning_rate=aux_learning_rate,
            l_bpp_weight=l_bpp_weight,
            l_guide_weight=l_guide_weight,
            sync_path=sync_path,
            synch_control=synch_control,
            ckpt_path_pre=ckpt_path_pre,
            preprocess_config=preprocess_config,
            calculate_metrics=calculate_metrics,
            first_stage_key=first_stage_key,
            *args,
            **kwargs
        )
        
        self.bg_encoder = MonitorBGEncoder(
            in_channels=3, out_channels=bg_encoder_out_channels, freeze=True
        )
        self.dbc = DynamicBackgroundCorrection(
            in_channels=bg_encoder_out_channels, delta_channels=dbc_delta_channels
        )
        self.cmf = InnovativeCrossModalFusion(
            img_dim=cmf_img_dim, txt_dim=cmf_txt_dim, num_heads=cmf_num_heads
        )
        self.txt_encoder = TextEncoder()

    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x_latent, cond = super().get_input(batch, self.first_stage_key, bs=bs, *args, **kwargs)

        control = cond['control'][0]
        # 修复：接受256x256的训练尺寸，同时支持512x512的验证尺寸
        assert control.shape[2:] in [(256, 256), (512, 512)], f"control尺寸错误：{control.shape[2:]}，期望(256,256)或(512,512)"
        
        z_bg_curr = self.bg_encoder(control)
        bg_prior = batch['bg']
        if bs is not None:
            bg_prior = bg_prior[:bs]
        bg_prior = bg_prior.to(self.device)
        bg_prior = einops.rearrange(bg_prior, 'b h w c -> b c h w')
        # 修复：接受256x256的训练尺寸，同时支持512x512的验证尺寸
        assert bg_prior.shape[2:] in [(256, 256), (512, 512)], f"bg_prior尺寸错误：{bg_prior.shape[2:]}，期望(256,256)或(512,512)"
        
        z_bg_prior = self.bg_encoder(bg_prior)
        delta_feat = self.dbc.encode(z_bg_prior, z_bg_curr)
        
        N, _, H_delta, W_delta = delta_feat.shape
        num_pixels_delta = N * H_delta * W_delta
        delta_bpp = (torch.log(torch.ones_like(delta_feat)).sum() / (-math.log(2) * num_pixels_delta))
        
        text_list = batch['text'][:bs] if bs is not None else batch['text']
        z_txt = self.txt_encoder(text_list)
        
        cond.update({
            'z_bg_prior': [z_bg_prior],
            'z_bg_curr': [z_bg_curr],
            'delta_feat': [delta_feat],
            'delta_bpp': delta_bpp,
            'z_txt': [z_txt],
            'bg_prior': [bg_prior],
            'x_latent': [x_latent]
        })
        del control, bg_prior, z_bg_curr, z_bg_prior, delta_feat, z_txt
        
        return x_latent, cond

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model
    
        cond_txt = torch.cat(cond['c_crossattn'], 1)
        cond_hint = torch.cat(cond['c_latent'], 1)
        
        z_bg_prior = cond['z_bg_prior'][0]
        delta_feat = cond['delta_feat'][0]
        z_txt = cond['z_txt'][0]
        
        z_bg_corrected = self.dbc.decode(delta_feat, z_bg_prior)
        # 断言：z_bg_corrected尺寸正确（64x64）
        assert z_bg_corrected.shape[2:] == (64, 64), f"z_bg_corrected尺寸错误：{z_bg_corrected.shape[2:]}"
        
        # 核心调用：现在所有张量尺寸都有断言保障
        z_fused = self.cmf(x_noisy, z_bg_corrected, z_txt)
        
        # 加权融合
        fusion_weight = 0.3
        cond_hint_fused = (1 - fusion_weight) * cond_hint + fusion_weight * z_fused
        
        eps = self.control_model(
            x=x_noisy, timesteps=t, context=cond_txt,
            hint=cond_hint_fused, base_model=diffusion_model)
        
        del cond_txt, cond_hint, z_bg_prior, delta_feat, z_txt, z_bg_corrected, z_fused, cond_hint_fused
        return eps

    def log_images(self, batch, sample_steps=50, bs=1):
        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=bs)
        bpp = c["q_bpp"]
        bpp_img = [f'{bpp:2f}']*4
        
        c_latent = c["c_latent"][0]
        control = c["control"][0]
        c_txt = c["c_crossattn"][0]
        
        bs_safe = bs if bs is not None else 1
        z_bg_prior = c.get(
            "z_bg_prior", 
            [torch.zeros((bs_safe, 128, 64, 64), device=self.device)]
        )[0]
        z_bg_curr = c.get("z_bg_curr", [torch.zeros_like(z_bg_prior)])[0]
        delta_feat = c.get(
            "delta_feat", 
            [torch.zeros((bs_safe, 8, 32, 32), device=self.device)]
        )[0]
        z_txt = c.get(
            "z_txt", 
            [torch.zeros((bs_safe, 768), device=self.device)]
        )[0]
        delta_bpp = c.get("delta_bpp", torch.tensor(0.0, device=self.device))
        bg_prior = c.get(
            "bg_prior", 
            [torch.zeros((bs_safe, 3, 256, 256), device=self.device)]
        )[0]
        x_latent = c.get("x_latent", [z])[0]

        log["hq"] = (self.decode_first_stage(z) + 1) / 2
        log["control"] = control
        log["text"] = (log_txt_as_img((512, 512), bpp_img, size=16) + 1) / 2
        
        full_cond = {
            "c_crossattn": [c_txt],
            "c_latent": [c_latent],
            "z_bg_prior": [z_bg_prior],
            "z_bg_curr": [z_bg_curr],
            "delta_feat": [delta_feat],
            "z_txt": [z_txt],
            "delta_bpp": delta_bpp,
            "bg_prior": [bg_prior],
            "x_latent": [x_latent],
            "bpp": c["bpp"],
            "q_bpp": c["q_bpp"]
        }
        
        samples = self.sample_log(cond=full_cond, steps=min(20, sample_steps))
        x_samples = self.decode_first_stage(samples)
        log["samples"] = (x_samples + 1) / 2
        
        log['bg_prior'] = bg_prior[:, :, ::4, ::4]
        text_list = batch['text'][:bs_safe] if bs is not None else batch['text']
        text_img = log_txt_as_img((256, 256), text_list, size=12)
        log['text_desc'] = (text_img + 1) / 2
        
        z_bg_corrected = self.dbc.decode(delta_feat, z_bg_prior)
        z_bg_corrected_vis = (z_bg_corrected - z_bg_corrected.min()) / (z_bg_corrected.max() - z_bg_corrected.min())
        log['bg_corrected'] = z_bg_corrected_vis[:, :3, ::1, ::1]
        
        del z, c, c_latent, control, c_txt, z_bg_prior, z_bg_curr, delta_feat, z_txt, delta_bpp, bg_prior, x_latent, samples, x_samples, z_bg_corrected, z_bg_corrected_vis, text_img
        torch.cuda.empty_cache()
        
        return log, bpp

    def configure_optimizers(self):
        opt, aux_opt = super().configure_optimizers()
        
        new_params = [
            p for p in list(self.dbc.parameters()) + list(self.cmf.parameters()) 
            if p.requires_grad
        ]
        opt.add_param_group({'params': new_params, 'lr': self.learning_rate})
        
        opt.defaults['maximize'] = False
        opt.defaults['foreach'] = False
        
        return opt, aux_opt

    def validation_step(self, batch, batch_idx):
        out = super().validation_step(batch, batch_idx)
        
        control = batch[self.control_key][batch_idx:batch_idx+1].to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        z_bg_curr = self.bg_encoder(control)
        
        bg_prior = batch['bg'][batch_idx:batch_idx+1].to(self.device)
        bg_prior = einops.rearrange(bg_prior, 'b h w c -> b c h w')
        z_bg_prior = self.bg_encoder(bg_prior)
        
        delta_feat = self.dbc.encode(z_bg_prior, z_bg_curr)
        z_bg_corrected = self.dbc.decode(delta_feat, z_bg_prior)
        
        bg_correction_mse = F.mse_loss(z_bg_corrected, z_bg_curr).cpu().numpy()
        out.append(bg_correction_mse)
        
        del control, z_bg_curr, bg_prior, z_bg_prior, delta_feat, z_bg_corrected
        torch.cuda.empty_cache()
        
        return out

    def p_losses(self, x_start, cond, t, noise=None):
        loss_dict = {}
        prefix = 'T' if self.training else 'V'

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond)

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/l_simple': loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/l_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_bpp = cond['bpp']
        guide_bpp = cond['q_bpp']
        loss_dict.update({f'{prefix}/l_bpp': loss_bpp.mean()})
        loss_dict.update({f'{prefix}/q_bpp': guide_bpp.mean()})
        loss += self.l_bpp_weight * loss_bpp

        c_latent = cond['c_latent'][0][:,:4,:,:]
        loss_guide = self.get_loss(c_latent, x_start)
        loss_dict.update({f'{prefix}/l_guide': loss_guide.mean()})
        loss += self.l_guide_weight * loss_guide

        z_bg_curr = cond['z_bg_curr'][0]
        z_bg_prior = cond['z_bg_prior'][0]
        delta_feat = cond['delta_feat'][0]
        z_bg_corrected = self.dbc.decode(delta_feat, z_bg_prior)
        loss_bg_correction = F.mse_loss(z_bg_corrected, z_bg_curr)
        loss_dict.update({f'{prefix}/l_bg_correction': loss_bg_correction.mean()})
        loss += 0.01 * loss_bg_correction

        z_img = x_start
        z_txt = cond['z_txt'][0]
        z_txt_align = self.cmf.txt_proj(z_txt).repeat(1, 1, z_img.shape[2], z_img.shape[3])
        z_img_flat = z_img.flatten(1)
        z_txt_flat = z_txt_align.flatten(1)
        cos_sim = F.cosine_similarity(z_img_flat, z_txt_flat, dim=-1).mean()
        loss_text_align = 1 - cos_sim
        loss_dict.update({f'{prefix}/l_text_align': loss_text_align.mean()})
        loss += 0.005 * loss_text_align

        delta_bpp = cond['delta_bpp']
        loss_dict.update({f'{prefix}/l_delta_bpp': delta_bpp.mean()})
        loss += 0.01 * delta_bpp

        loss_dict.update({f'{prefix}/loss': loss})
        
        del z_bg_curr, z_bg_prior, delta_feat, z_bg_corrected, z_img, z_txt, z_txt_align, z_img_flat, z_txt_flat
        torch.cuda.empty_cache()
        
        return loss, loss_dict
    
    @torch.no_grad()
    def sample_log(self, cond, steps):
        sampler = SpacedSampler(self)
        b, c, h, w = cond["c_latent"][0].shape
        shape = (b, self.channels, 64, 64)

        if 'z_bg_prior' not in cond:
            device = cond["c_latent"][0].device
            z_bg_prior = torch.zeros((b, 128, 64, 64), device=device)
            z_bg_curr = torch.zeros_like(z_bg_prior)
            delta_feat = torch.zeros((b, 8, 32, 32), device=device)
            z_txt = torch.zeros((b, 768), device=device)
            
            cond.update({
                'z_bg_prior': [z_bg_prior],
                'z_bg_curr': [z_bg_curr],
                'delta_feat': [delta_feat],
                'z_txt': [z_txt],
                'delta_bpp': torch.tensor(0.0, device=device),
                'bg_prior': [torch.zeros((b, 3, 256, 256), device=device)],
                'x_original': [torch.zeros((b, 3, 256, 256), device=device)],
                'x_latent': [torch.zeros((b, self.channels, 64, 64), device=device)]
            })

        samples = sampler.sample(
            steps, shape, cond, unconditional_guidance_scale=1.0,
            unconditional_conditioning=None
        )
        samples = F.interpolate(samples, size=(h, w), mode='bilinear', align_corners=False)
        return samples