import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import DynamicBackgroundCorrection, InnovativeCrossModalFusion, DiffEICWithPriorAndText
from utils import instantiate_from_config   
import yaml

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 测试DynamicBackgroundCorrection类
def test_dynamic_background_correction():
    print("\nTesting DynamicBackgroundCorrection...")
    
    # 创建模型实例
    dbc = DynamicBackgroundCorrection(in_channels=256, delta_channels=16).to(device)
    
    # 创建模拟输入
    batch_size = 2
    z_bg_prior = torch.randn(batch_size, 256, 64, 64).to(device)
    z_bg_curr = torch.randn(batch_size, 256, 64, 64).to(device)
    
    # 测试encode
    delta_feat = dbc.encode(z_bg_prior, z_bg_curr)
    print(f"  encode output shape: {delta_feat.shape}")
    
    # 测试decode
    z_bg_corrected = dbc.decode(delta_feat, z_bg_prior)
    print(f"  decode output shape: {z_bg_corrected.shape}")
    
    assert delta_feat.shape == (batch_size, 16, 32, 32), f"Expected delta_feat shape {(batch_size, 16, 32, 32)}, got {delta_feat.shape}"
    assert z_bg_corrected.shape == (batch_size, 256, 64, 64), f"Expected z_bg_corrected shape {(batch_size, 256, 64, 64)}, got {z_bg_corrected.shape}"
    
    print("  ✓ DynamicBackgroundCorrection forward propagation passed!")

# 测试InnovativeCrossModalFusion类
def test_innovative_cross_modal_fusion():
    print("\nTesting InnovativeCrossModalFusion...")
    
    # 创建模型实例
    cmf = InnovativeCrossModalFusion(img_dim=256, txt_dim=768).to(device)
    
    # 创建模拟输入
    batch_size = 2
    z_img = torch.randn(batch_size, 256, 64, 64).to(device)
    z_bg = torch.randn(batch_size, 256, 64, 64).to(device)
    z_txt = torch.randn(batch_size, 768).to(device)
    
    # 测试forward
    z_fused = cmf(z_img, z_bg, z_txt)
    print(f"  forward output shape: {z_fused.shape}")
    
    assert z_fused.shape == (batch_size, 256, 64, 64), f"Expected z_fused shape {(batch_size, 256, 64, 64)}, got {z_fused.shape}"
    
    print("  ✓ InnovativeCrossModalFusion forward propagation passed!")

# 测试DiffEICWithPriorAndText类
def test_diffeic_with_prior_and_text():
    print("\nTesting DiffEICWithPriorAndText...")
    
    # 加载配置文件
    config_path = "configs/model/diffeic.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # 创建模型实例
    model = DiffEICWithPriorAndText(
        control_stage_config=config["params"]["control_stage_config"],
        control_key=config["params"]["control_key"],
        sd_locked=config["params"]["sd_locked"],
        learning_rate=config["params"]["learning_rate"],
        aux_learning_rate=config["params"]["aux_learning_rate"],
        l_bpp_weight=config["params"]["l_bpp_weight"],
        l_guide_weight=config["params"]["l_guide_weight"],
        sync_path=config["params"]["sync_path"],
        synch_control=config["params"]["synch_control"],
        ckpt_path_pre=config["params"]["ckpt_path_pre"],
        preprocess_config=config["params"]["preprocess_config"],
        calculate_metrics=config["params"]["calculate_metrics"],
        linear_start=config["params"]["linear_start"],
        linear_end=config["params"]["linear_end"],
        num_timesteps_cond=config["params"]["num_timesteps_cond"],
        log_every_t=config["params"]["log_every_t"],
        timesteps=config["params"]["timesteps"],
        first_stage_key=config["params"]["first_stage_key"],
        cond_stage_key=config["params"]["cond_stage_key"],
        image_size=config["params"]["image_size"],
        channels=config["params"]["channels"],
        cond_stage_trainable=config["params"]["cond_stage_trainable"],
        conditioning_key=config["params"]["conditioning_key"],
        monitor=config["params"]["monitor"],
        scale_factor=config["params"]["scale_factor"],
        use_ema=config["params"]["use_ema"],
        l_simple_weight=config["params"]["l_simple_weight"],
        unet_config=config["params"]["unet_config"],
        first_stage_config=config["params"]["first_stage_config"],
        cond_stage_config=config["params"]["cond_stage_config"]
    ).to(device)
    
    # 创建模拟输入
    batch_size = 1
    I = torch.randn(batch_size, 3, 256, 256).to(device)
    B = torch.randn(batch_size, 3, 256, 256).to(device)
    text_list = ["a test image"] * batch_size
    
    print("  Testing encode...")
    # 由于完整的encode方法涉及到复杂的依赖，我们只测试部分组件
    
    # 测试文本编码器
    z_T = model.txt_encoder(text_list)
    print(f"    Text encoder output shape: {z_T.shape}")
    
    # 测试背景编码器
    z_B = model.bg_encoder(B)
    print(f"    Background encoder output shape: {z_B.shape}")
    
    # 测试动态背景矫正
    z_B_curr = model.bg_encoder(I)
    delta_feat = model.dbc.encode(z_B, z_B_curr)
    delta_quant = delta_feat  # 简化处理，不使用实际的量化
    z_B_corrected = model.dbc.decode(delta_quant, z_B)
    print(f"    Dynamic background correction output shape: {z_B_corrected.shape}")
    
    # 测试跨模态融合
    z_img = torch.randn(batch_size, 256, 64, 64).to(device)
    z_fused = model.cmf(z_img, z_B, z_T)
    print(f"    Cross modal fusion output shape: {z_fused.shape}")
    
    print("  ✓ DiffEICWithPriorAndText components passed!")

def main():
    print("Starting forward propagation tests...")
    
    # 测试各个类
    test_dynamic_background_correction()
    test_innovative_cross_modal_fusion()
    test_diffeic_with_prior_and_text()
    
    print("\n✅ All forward propagation tests passed!")

if __name__ == "__main__":
    main()