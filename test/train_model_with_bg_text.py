import os
import sys
import torch
from argparse import ArgumentParser
from omegaconf import OmegaConf

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.common import instantiate_from_config, load_state_dict


def main() -> None:
    # 解析命令行参数
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default='./configs/train_diffeic_with_bg_and_text.yaml')
    args = parser.parse_args()
    
    # 加载主配置文件
    config = OmegaConf.load(args.config)
    
    # 设置随机种子
    import pytorch_lightning as pl
    pl.seed_everything(config.lightning.seed, workers=True)
    
    # 实例化数据模块
    data_module = instantiate_from_config(config.data)
    
    # 加载模型配置并实例化模型
    model = instantiate_from_config(OmegaConf.load(config.model.config))
    
    # 恢复训练（如果指定了检查点路径）
    if config.model.get("resume"):
        load_state_dict(model, torch.load(config.model.resume, map_location="cpu"), strict=True)
    
    # 实例化回调
    callbacks = []
    for callback_config in config.lightning.callbacks:
        callbacks.append(instantiate_from_config(callback_config))
    
    # 实例化训练器
    trainer = pl.Trainer(callbacks=callbacks, **config.lightning.trainer)
    
    # 开始训练
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()