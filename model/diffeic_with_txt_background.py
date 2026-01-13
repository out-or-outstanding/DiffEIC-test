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


class DynamicBackgroundCorrection(nn.Module):
    def __init__(self, in_channels, delta_channels=16):
        super().__init__()
        self.in_channels = in_channels
        # 编码端：新增1x1卷积先对齐特征分布（适配自主bg_encoder的输出）
        self.encoder_delta = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, padding=0),  # 新增：对齐特征分布
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels, delta_channels*2, 3, padding=1),
            nn.BatchNorm2d(delta_channels*2),
            nn.LeakyReLU(),
            nn.Conv2d(delta_channels*2, delta_channels, 3, padding=1, stride=2),
            nn.BatchNorm2d(delta_channels),
            nn.LeakyReLU()
        )
        # 解码端：保持不变，仅适配尺寸
        self.decoder_delta = nn.Sequential(
            nn.ConvTranspose2d(delta_channels, delta_channels*2, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(delta_channels*2),
            nn.LeakyReLU(),
            nn.Conv2d(delta_channels*2, in_channels, 3, padding=1),
            nn.Tanh()
        )

    def encode(self, z_bg_prior, z_bg_curr):
        # 新增：对差异做归一化，避免数值波动过大
        delta_raw = z_bg_curr - z_bg_prior
        delta_raw = (delta_raw - delta_raw.mean(dim=[2,3], keepdim=True)) / (delta_raw.std(dim=[2,3], keepdim=True) + 1e-6)
        delta_feat = self.encoder_delta(delta_raw)
        return delta_feat

    def decode(self, delta_quant, z_bg_prior):
        delta_raw_rec = self.decoder_delta(delta_quant)
        delta_raw_rec = F.interpolate(
            delta_raw_rec, size=z_bg_prior.shape[2:], 
            mode='bilinear', align_corners=False
        )
        # 新增：限制矫正幅度，避免过矫正
        delta_raw_rec = torch.clamp(delta_raw_rec, -0.5, 0.5)
        z_bg_corrected = z_bg_prior + delta_raw_rec
        return z_bg_corrected


class BiDirectionalCrossAttention(nn.Module):
    """双向交叉注意力层（核心创新组件）"""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        # 自动调整num_heads使其能被dim整除
        if dim < num_heads:
            # 如果dim小于num_heads，将num_heads设为dim
            num_heads = dim
        elif dim % num_heads != 0:
            # 如果dim不能被num_heads整除，找到最大的能被dim整除的num_heads
            for i in range(num_heads, 0, -1):
                if dim % i == 0:
                    num_heads = i
                    break
        
        self.num_heads = num_heads
        self.dim = dim
        self.scale = (dim // num_heads) ** -0.5
        # 注意力投影层（共享）
        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, q, kv):
        """
        q: 查询特征（如图像特征） [B, C, H, W]
        kv: 键值特征（如背景+文本拼接） [B, C, H*2, W]
        """
        B, C, H, W = q.shape
        
        # 检查输入维度是否匹配
        if C != self.dim:
            # 如果维度不匹配，使用1x1卷积进行维度对齐
            if not hasattr(self, 'dim_align'):
                self.dim_align = nn.Conv2d(C, self.dim, 1).to(q.device)
            q = self.dim_align(q)
            kv = self.dim_align(kv)
            C = self.dim
        
        # 展平空间维度：[B, C, H, W] → [B, H*W, C]
        q_flat = q.permute(0, 2, 3, 1).reshape(B, -1, C)
        kv_flat = kv.permute(0, 2, 3, 1).reshape(B, -1, C)
        
        # 多头注意力计算
        # 处理查询q
        qkv_q = self.qkv_proj(q_flat)
        q, k_q, v_q = qkv_q.chunk(3, dim=-1)
        q = q.reshape(B, -1, self.num_heads, C//self.num_heads).transpose(1, 2)
        
        # 处理键值kv
        qkv_kv = self.qkv_proj(kv_flat)
        k_kv, k, v = qkv_kv.chunk(3, dim=-1)
        k = k.reshape(B, -1, self.num_heads, C//self.num_heads).transpose(1, 2)
        v = v.reshape(B, -1, self.num_heads, C//self.num_heads).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        out = self.out_proj(out)
        # 恢复空间维度
        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return out

        

class InnovativeCrossModalFusion(nn.Module):
    def __init__(self, img_dim, txt_dim=768, num_heads=8):
        super().__init__()
        self.img_dim = img_dim
        # 维度对齐（文本→图像维度，背景复用原维度）
        self.txt_proj = nn.Sequential(
            nn.Linear(txt_dim, img_dim),
            nn.LeakyReLU(),
            nn.Unflatten(1, (img_dim, 1, 1))  # [B, D] → [B, D, 1, 1]
        )
        # 双向交叉注意力
        self.attn_img2bg_txt = BiDirectionalCrossAttention(img_dim, num_heads)
        self.attn_bg_txt2img = BiDirectionalCrossAttention(img_dim, num_heads)
        # 模态门控
        self.gating = ModalityAwareGating(img_dim)
        # 残差校准（保留原图像特征）
        self.res_calibrate = nn.Conv2d(img_dim, img_dim, 1)

    def forward(self, z_img, z_bg, z_txt):
        """
        输入：
        - z_img: 图像潜在特征 [B, C, H, W]
        - z_bg: 矫正后背景特征 [B, C, H, W]
        - z_txt: CLIP文本特征 [B, 768]
        输出：融合特征 [B, C, H, W]
        """
        B, C_img, H_img, W_img = z_img.shape
        B, C_bg, H_bg, W_bg = z_bg.shape
        
        # 1. 维度对齐：确保所有输入特征维度一致
        if C_img != self.img_dim:
            if not hasattr(self, 'img_align'):
                self.img_align = nn.Conv2d(C_img, self.img_dim, 1).to(z_img.device)
            z_img = self.img_align(z_img)
        
        if C_bg != self.img_dim:
            if not hasattr(self, 'bg_align'):
                self.bg_align = nn.Conv2d(C_bg, self.img_dim, 1).to(z_bg.device)
            z_bg = self.bg_align(z_bg)
        
        # 2. 文本特征空间对齐
        z_txt_align = self.txt_proj(z_txt)
        z_txt_align = z_txt_align.repeat(1, 1, H_img, W_img)
        
        # 3. 双向注意力计算
        # 图像→背景+文本
        kv = torch.cat([z_bg, z_txt_align], dim=2)  # 拼接空间维度
        attn_img = self.attn_img2bg_txt(z_img, kv)
        # 背景+文本→图像
        q_bg_txt = torch.cat([z_bg, z_txt_align], dim=2)
        attn_bg_txt = self.attn_bg_txt2img(q_bg_txt, z_img)
        attn_bg, attn_txt = torch.split(attn_bg_txt, z_bg.shape[2], dim=2)  # 拆分背景/文本注意力
        
        # 4. 自适应加权融合
        weights = self.gating(z_img)  # [B, 3] → α, β, γ
        α, β, γ = [w.reshape(-1, 1, 1, 1) for w in weights.chunk(3, dim=1)]
        z_fused = α * attn_img + β * (z_bg + attn_bg) + γ * (z_txt_align + attn_txt)
        
        # 5. 残差校准（兼容原预训练权重）
        z_fused = self.res_calibrate(z_fused) + z_img
        return z_fused


class ModalityAwareGating(nn.Module):
    """模态感知门控（自适应权重）"""
    def __init__(self, dim):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim//4),
            nn.LeakyReLU(),
            nn.Linear(dim//4, 3),  # 输出：图像α、背景β、文本γ（和为1）
            nn.Softmax(dim=-1)
        )

    def forward(self, z_img):
        B, C = z_img.shape[:2]
        feat_gap = self.gap(z_img).reshape(B, C)
        return self.mlp(feat_gap)  # [B, 3]


class InnovativeCrossModalFusion(nn.Module):
    def __init__(self, img_dim, txt_dim=768, num_heads=8):
        super().__init__()
        self.img_dim = img_dim
        # 维度对齐（文本→图像维度，背景复用原维度）
        self.txt_proj = nn.Sequential(
            nn.Linear(txt_dim, img_dim),
            nn.LeakyReLU(),
            nn.Unflatten(1, (img_dim, 1, 1))  # [B, D] → [B, D, 1, 1]
        )
        # 双向交叉注意力
        self.attn_img2bg_txt = BiDirectionalCrossAttention(img_dim, num_heads)
        self.attn_bg_txt2img = BiDirectionalCrossAttention(img_dim, num_heads)
        # 模态门控
        self.gating = ModalityAwareGating(img_dim)
        # 残差校准（保留原图像特征）
        self.res_calibrate = nn.Conv2d(img_dim, img_dim, 1)

    def forward(self, z_img, z_bg, z_txt):
        """
        输入：
        - z_img: 图像潜在特征 [B, C, H, W]
        - z_bg: 矫正后背景特征 [B, C, H, W]
        - z_txt: CLIP文本特征 [B, 768]
        输出：融合特征 [B, C, H, W]
        """
        B, C_img, H_img, W_img = z_img.shape
        B, C_bg, H_bg, W_bg = z_bg.shape
        
        # 1. 维度对齐：确保所有输入特征维度一致
        if C_img != self.img_dim:
            if not hasattr(self, 'img_align'):
                self.img_align = nn.Conv2d(C_img, self.img_dim, 1).to(z_img.device)
            z_img = self.img_align(z_img)
        
        if C_bg != self.img_dim:
            if not hasattr(self, 'bg_align'):
                self.bg_align = nn.Conv2d(C_bg, self.img_dim, 1).to(z_bg.device)
            z_bg = self.bg_align(z_bg)
        
        # 2. 文本特征空间对齐
        z_txt_align = self.txt_proj(z_txt)
        z_txt_align = z_txt_align.repeat(1, 1, H_img, W_img)
        
        # 3. 双向注意力计算
        # 图像→背景+文本
        kv = torch.cat([z_bg, z_txt_align], dim=2)  # 拼接空间维度
        attn_img = self.attn_img2bg_txt(z_img, kv)
        # 背景+文本→图像
        q_bg_txt = torch.cat([z_bg, z_txt_align], dim=2)
        attn_bg_txt = self.attn_bg_txt2img(q_bg_txt, z_img)
        attn_bg, attn_txt = torch.split(attn_bg_txt, z_bg.shape[2], dim=2)  # 拆分背景/文本注意力
        
        # 4. 自适应加权融合
        weights = self.gating(z_img)  # [B, 3] → α, β, γ
        α, β, γ = [w.reshape(-1, 1, 1, 1) for w in weights.chunk(3, dim=1)]
        z_fused = α * attn_img + β * (z_bg + attn_bg) + γ * (z_txt_align + attn_txt)
        
        # 5. 残差校准（兼容原预训练权重）
        z_fused = self.res_calibrate(z_fused) + z_img
        return z_fused


class MonitorBGEncoder(nn.Module):
    """
    自主设计的背景特征提取器（监控场景优化版）
    新增：多尺度特征融合 + 扰动感知卷积，适配背景轻微变化的特点
    """
    def __init__(self, in_channels=3, out_channels=256, freeze=True):
        super().__init__()
        # 主干卷积（基础特征提取）
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 多尺度特征融合（捕捉不同尺度的背景扰动）
        self.fusion = nn.Conv2d(out_channels + 128, out_channels, 1, stride=1, padding=0)
        
        # 扰动感知卷积（增强对微小变化的敏感度）
        self.disturb_conv = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=2, dilation=2)
        
        # 冻结权重
        if freeze:
            for param in self.parameters():
                param.requires_grad = False
        
        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m == self.disturb_conv:  # 扰动感知卷积用特殊初始化
                    nn.init.normal_(m.weight, mean=0, std=0.01)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 1. 基础特征提取
        feat1 = self.conv1(x)
        feat2 = self.conv2(feat1)
        feat3 = self.conv3(feat2)
        
        # 2. 多尺度融合（feat2上采样到feat3尺寸）
        feat2_up = F.interpolate(feat2, size=feat3.shape[2:], mode='bilinear', align_corners=False)
        feat_fused = self.fusion(torch.cat([feat3, feat2_up], dim=1))
        
        # 3. 扰动感知增强
        z_bg = self.disturb_conv(feat_fused) + feat3  # 残差连接，保留原特征
        
        return z_bg


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
        self.clip_txt = CLIPTextModel.from_pretrained("./weight/clip_vit_base_patch16")
        # 冻结CLIP权重（仅投影层微调）
        for param in self.clip_txt.parameters():
            param.requires_grad = False

    def forward(self, text_list):
        """文本→768维特征"""
        tokens = self.tokenizer(
            text_list, return_tensors="pt", padding=True, truncation=True
        ).to(next(self.parameters()).device)
        return self.clip_txt(**tokens).pooler_output  # [B, 768]


# 核心：整合所有模块的DiffEICWithPriorAndText类
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
        # 新增参数：背景/文本模块配置
        bg_encoder_out_channels: int = 256,
        dbc_delta_channels: int = 16,
        cmf_img_dim: int = 4,
        cmf_num_heads: int = 8,
        first_stage_key: str = 'jpg',  # 显式指定first_stage_key为3通道图像的key
        *args, 
        **kwargs
    ) -> "DiffEICWithPriorAndText":
        # 调用父类DiffEIC的初始化（保留所有核心逻辑）
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
            first_stage_key=first_stage_key,  # 传递给父类
            *args,
            **kwargs
        )
        
        # 初始化自定义子模块（核心新增）
        # 1. 背景特征提取器（监控场景优化版）
        self.bg_encoder = MonitorBGEncoder(
            in_channels=3,  # 输入为3通道RGB背景图
            out_channels=bg_encoder_out_channels,
            freeze=True  # 默认冻结，如需微调设为False
        )
        
        # 2. 动态背景矫正模块
        self.dbc = DynamicBackgroundCorrection(
            in_channels=bg_encoder_out_channels,
            delta_channels=dbc_delta_channels
        )
        
        # 3. 跨模态融合模块（图像+背景+文本）
        self.cmf = InnovativeCrossModalFusion(
            img_dim=cmf_img_dim,
            txt_dim=512,  # 恢复为CLIP原生512维（无需适配512）
            num_heads=cmf_num_heads
        )
        
        # 4. 文本编码器（复用CLIP）
        self.txt_encoder = TextEncoder()

    def get_input(self, batch, k, bs=None, *args, **kwargs):
        """
        保留父类逻辑，新增背景/文本特征处理
        batch字段要求：
        - batch['jpg']: 原始3通道RGB图像 [B, 3, H, W]
        - batch['bg']: 先验背景图 [B, H, W, 3]
        - batch['text']: 文本描述列表 [B]
        """
        # 调用父类get_input：处理3通道原始图像，返回512通道潜在特征x_latent
        x_latent, cond = super().get_input(batch, self.first_stage_key, bs=bs, *args, **kwargs)

        # 处理背景特征（先验背景 + 当前背景）
        # 提取当前背景（control即为当前帧的背景区域，数据集已修复为3通道）
        control = cond['control'][0]  # [B, 3, H, W]
        z_bg_curr = self.bg_encoder(control)  # 当前背景特征 [B, 256, H//4, W//4]
        
        # 提取先验背景（从batch中获取，数据集已修复为3通道）
        bg_prior = batch['bg']  # [B, H, W, 3]
        if bs is not None:
            bg_prior = bg_prior[:bs]
        bg_prior = bg_prior.to(self.device)
        bg_prior = einops.rearrange(bg_prior, 'b h w c -> b c h w').float()
        z_bg_prior = self.bg_encoder(bg_prior)  # 先验背景特征 [B, 256, H//4, W//4]
        
        # 动态背景矫正：编码背景差异（加入码率计算）
        delta_feat = self.dbc.encode(z_bg_prior, z_bg_curr)  # [B, 16, H//8, W//8]
        # 计算背景差异的bpp（和原bpp逻辑对齐）
        N, _, H_delta, W_delta = delta_feat.shape
        num_pixels_delta = N * H_delta * W_delta
        delta_bpp = (torch.log(torch.ones_like(delta_feat)).sum() / (-math.log(2) * num_pixels_delta))
        
        # 处理文本特征
        text_list = batch['text']  # [B]
        if bs is not None:
            text_list = text_list[:bs]
        z_txt = self.txt_encoder(text_list)  # [B, 768]
        
        # 更新cond字典，新增背景/文本特征
        cond.update({
            'z_bg_prior': [z_bg_prior],
            'z_bg_curr': [z_bg_curr],
            'delta_feat': [delta_feat],
            'delta_bpp': delta_bpp,
            'z_txt': [z_txt],
            'bg_prior': [bg_prior],
            'x_latent': [x_latent]  # 添加x_latent字段
        })

        # 保留父类返回的x（潜在特征），返回最终结果
        return x_latent, cond

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        """
        重写apply_model：加入背景矫正 + 跨模态融合，再传入control_model
        """
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        # 1. 提取原有的文本/控制特征
        cond_txt = torch.cat(cond['c_crossattn'], 1)
        cond_hint = torch.cat(cond['c_latent'], 1)
        
        # 2. 提取背景/文本特征（从cond中获取）
        z_bg_prior = cond['z_bg_prior'][0]
        delta_feat = cond['delta_feat'][0]
        z_txt = cond['z_txt'][0]
        
        # 3. 动态背景矫正：解码差异，得到矫正后的背景特征
        z_bg_corrected = self.dbc.decode(delta_feat, z_bg_prior)
        
        # 4. 跨模态融合：图像噪声特征 + 矫正背景 + 文本
        # 将x_noisy调整到和背景特征相同的维度（下采样4倍）
        x_noisy_down = F.interpolate(
            x_noisy, size=z_bg_corrected.shape[2:], mode='bilinear', align_corners=False
        )
        # 跨模态融合
        z_fused = self.cmf(x_noisy_down, z_bg_corrected, z_txt)
        
        # 5. 将融合特征上采样回原尺寸，加入到control_model的hint中
        z_fused_up = F.interpolate(
            z_fused, size=x_noisy.shape[2:], mode='bilinear', align_corners=False
        )
        cond_hint_fused = torch.cat([cond_hint, z_fused_up], dim=1)  # 拼接融合特征
        
        # 6. 调用control_model（复用原逻辑，传入融合后的hint）
        eps = self.control_model(
            x=x_noisy, timesteps=t, context=cond_txt, hint=cond_hint_fused, base_model=diffusion_model)
        
        return eps

    def log_images(self, batch, sample_steps=50, bs=2):
        """
        重写log_images：确保在采样时传入完整的条件信息
        """
        log = dict()
        # 调用get_input获取完整的条件信息
        z, c = self.get_input(batch, self.first_stage_key, bs=bs)
        bpp = c["q_bpp"]
        bpp_img = [f'{bpp:2f}']*4
        
        # 提取所有必要的条件字段
        c_latent = c["c_latent"][0]
        control = c["control"][0]
        c_txt = c["c_crossattn"][0]
        
        # 提取自定义字段（使用get方法避免KeyError）
        z_bg_prior = c.get("z_bg_prior", [torch.zeros_like(c_latent)])[0]
        z_bg_curr = c.get("z_bg_curr", [torch.zeros_like(c_latent)])[0]
        delta_feat = c.get("delta_feat", [torch.zeros_like(c_latent)])[0]
        z_txt = c.get("z_txt", [torch.zeros(c_latent.shape[0], 768, device=c_latent.device)])[0]
        delta_bpp = c.get("delta_bpp", torch.tensor(0.0, device=c_latent.device))
        bg_prior = c.get("bg_prior", [torch.zeros_like(control)])[0]
        x_latent = c.get("x_latent", [z])[0]  # 使用z作为默认值

        log["hq"] = (self.decode_first_stage(z) + 1) / 2
        log["control"] = control
        log["text"] = (log_txt_as_img((512, 512), bpp_img, size=16) + 1) / 2
        
        # 构建完整的条件字典
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
        
        # 调用sample_log，传入完整的条件信息
        samples = self.sample_log(cond=full_cond, steps=sample_steps)
        x_samples = self.decode_first_stage(samples)
        log["samples"] = (x_samples + 1) / 2
        
        # 新增背景/文本相关日志
        log['bg_prior'] = bg_prior  # 先验背景可视化
        
        # 提取文本描述并转为图像日志
        text_list = batch['text'][:bs] if bs is not None else batch['text']
        text_img = log_txt_as_img((512, 512), text_list, size=16)
        log['text_desc'] = (text_img + 1) / 2  # 文本描述可视化
        
        # 背景矫正后的特征可视化（转为图像）
        z_bg_corrected = self.dbc.decode(delta_feat, z_bg_prior)
        z_bg_corrected_vis = (z_bg_corrected - z_bg_corrected.min()) / (z_bg_corrected.max() - z_bg_corrected.min())
        log['bg_corrected'] = z_bg_corrected_vis[:, :3, :, :]  # 取前3通道可视化
        
        return log, bpp

    def configure_optimizers(self):
        """
        重写优化器配置：加入自定义模块的参数
        """
        # 调用父类优化器配置，获取原有参数
        opt, aux_opt = super().configure_optimizers()
        
        # 新增自定义模块的参数（如需微调背景编码器，需解冻）
        new_params = list(self.bg_encoder.parameters()) + \
                     list(self.dbc.parameters()) + \
                     list(self.cmf.parameters()) + \
                     list(self.txt_encoder.parameters())
        
        # 将新参数加入主优化器（保留原学习率）
        opt.add_param_group({'params': new_params, 'lr': self.learning_rate})
        
        return opt, aux_opt

    def validation_step(self, batch, batch_idx):
        """
        重写validation_step：新增背景/文本相关指标
        """
        # 调用父类validation_step，获取原有输出
        out = super().validation_step(batch, batch_idx)
        
        # 新增背景矫正效果指标（MSE：矫正后背景 vs 当前背景）
        control = batch[self.control_key][batch_idx:batch_idx+1]  # 修正索引，避免越界
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w').float()
        z_bg_curr = self.bg_encoder(control)
        
        bg_prior = batch['bg'][batch_idx:batch_idx+1]
        bg_prior = bg_prior.to(self.device)
        bg_prior = einops.rearrange(bg_prior, 'b h w c -> b c h w').float()
        z_bg_prior = self.bg_encoder(bg_prior)
        
        delta_feat = self.dbc.encode(z_bg_prior, z_bg_curr)
        z_bg_corrected = self.dbc.decode(delta_feat, z_bg_prior)
        
        bg_correction_mse = F.mse_loss(z_bg_corrected, z_bg_curr).cpu().numpy()
        out.append(bg_correction_mse)
        
        return out

    def validation_epoch_end(self, outputs: List):
        """
        重写validation_epoch_end：记录背景矫正指标
        """
        # 调用父类逻辑，记录原有指标
        super().validation_epoch_end(outputs)
        
        # 计算并记录背景矫正MSE
        outputs = np.array(outputs)
        avg_bg_correction_mse = sum(outputs[:, -1]) / len(outputs)
        self.log(
            "avg_bg_correction_mse", avg_bg_correction_mse,
            prog_bar=True, logger=True, on_step=False, on_epoch=True
        )

    def p_losses(self, x_start, cond, t, noise=None):
        loss_dict = {}
        prefix = 'T' if self.training else 'V'

        # ---------------------- 保留原损失计算逻辑 ----------------------
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

        # ---------------------- 新增自定义损失项 ----------------------
        # 1. 背景矫正损失：矫正后背景特征 vs 当前背景特征（MSE）
        z_bg_curr = cond['z_bg_curr'][0]
        z_bg_prior = cond['z_bg_prior'][0]
        delta_feat = cond['delta_feat'][0]
        z_bg_corrected = self.dbc.decode(delta_feat, z_bg_prior)
        loss_bg_correction = F.mse_loss(z_bg_corrected, z_bg_curr)
        loss_dict.update({f'{prefix}/l_bg_correction': loss_bg_correction.mean()})
        loss += 0.1 * loss_bg_correction  # 权重0.1，避免覆盖原损失

        # 2. 文本-图像对齐损失：融合特征中的文本分支 vs 图像特征（余弦相似度）
        z_img = x_start  # 原始图像特征
        z_txt = cond['z_txt'][0]
        # 文本特征空间对齐
        z_txt_align = self.cmf.txt_proj(z_txt)
        z_txt_align = z_txt_align.repeat(1, 1, z_img.shape[2], z_img.shape[3])
        # 展平计算余弦相似度损失
        z_img_flat = z_img.flatten(1)
        z_txt_flat = z_txt_align.flatten(1)
        cos_sim = F.cosine_similarity(z_img_flat, z_txt_flat, dim=-1).mean()
        loss_text_align = 1 - cos_sim  # 余弦相似度损失（1 - 相似度）
        loss_dict.update({f'{prefix}/l_text_align': loss_text_align.mean()})
        loss += 0.05 * loss_text_align  # 权重0.05，轻量级约束

        # 3. 背景差异bpp损失（和原bpp逻辑对齐）
        delta_bpp = cond['delta_bpp']
        loss_dict.update({f'{prefix}/l_delta_bpp': delta_bpp.mean()})
        loss += 0.01 * delta_bpp  # 小权重约束背景差异的码率

        # ---------------------- 保留原日志和返回 ----------------------
        loss_dict.update({f'{prefix}/loss': loss})
        return loss, loss_dict
    
    @torch.no_grad()
    def sample_log(self, cond, steps):
        """
        重写sample_log：确保在采样时传入完整的条件信息
        """
        sampler = SpacedSampler(self)
        b, c, h, w = cond["c_latent"][0].shape
        shape = (b, self.channels, h, w)

        # 确保cond包含所有必要的字段
        if 'z_bg_prior' not in cond:
            # 如果缺少自定义字段，使用默认值（零张量）填充
            device = cond["c_latent"][0].device
            z_bg_prior = torch.zeros(b, 256, h//4, w//4, device=device)
            z_bg_curr = torch.zeros(b, 256, h//4, w//4, device=device)
            delta_feat = torch.zeros(b, 16, h//8, w//8, device=device)
            z_txt = torch.zeros(b, 512, device=device)
            
            cond.update({
                'z_bg_prior': [z_bg_prior],
                'z_bg_curr': [z_bg_curr],
                'delta_feat': [delta_feat],
                'z_txt': [z_txt],
                'delta_bpp': torch.tensor(0.0, device=device),
                'bg_prior': [torch.zeros(b, 3, h, w, device=device)],
                'x_original': [torch.zeros(b, 3, h, w, device=device)],
                'x_latent': [torch.zeros(b, self.channels, h, w, device=device)]
            })

        samples = sampler.sample(
            steps, shape, cond, unconditional_guidance_scale=1.0,
            unconditional_conditioning=None
        )
        return samples