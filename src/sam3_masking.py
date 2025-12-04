#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAM3 分割模块
"""

import sys
import os
import torch
import numpy as np
from PIL import Image

# 使用正确的 SAM3 导入
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


class SAM3Predictor:
    """
    SAM3 预测器，用于图像分割
    """
    
    def __init__(self, checkpoint_path=None, device='cuda'):
        """
        初始化 SAM3
        
        Args:
            checkpoint_path: SAM3 模型路径
            device: 设备
        """
        self.device = device
        
        print("Loading SAM3...")
        
        # 构建 SAM3 模型
        # build_sam3_image_model 会自动处理 checkpont 下载和加载
        if checkpoint_path is None:
            checkpoint_path = "/home/zmh/SAM3-Video-Editor/pretrained_models/sam3/sam3.pt"
        self.model = build_sam3_image_model(
            device=device,
            checkpoint_path=checkpoint_path
        )
        
        # 强制移动到设备并设为 eval 模式
        self.model.to(device)
        self.model.eval()
        
        self.processor = Sam3Processor(self.model)
        
        print("SAM3 loaded successfully!")
    
    def generate_mask(self, image, text_prompt):
        """
        根据文本提示生成 mask
        
        Args:
            image: PIL.Image
            text_prompt: str，文本提示
        
        Returns:
            mask: torch.Tensor (1, 1, H, W)
        """
        # SAM3 需要 PIL Image
        
        # 设置图像
        inference_state = self.processor.set_image(image)
        
        # 使用文本提示生成 mask
        output = self.processor.set_text_prompt(
            state=inference_state,
            prompt=text_prompt
        )
        
        # 获取 masks
        # output["masks"] shape: (N, H, W)
        masks = output["masks"]
        
        if len(masks) > 0:
            # 选择第一个 mask
            mask = masks[0]
        else:
            # 如果没有检测到，返回全黑 mask
            w, h = image.size
            mask = torch.zeros((h, w), dtype=torch.float32, device=self.device)
        
        # 确保 mask 是 tensor 并且在正确的设备上
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).to(self.device)
            
        # 调整形状为 (1, 1, H, W)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(0)
            
        return mask.float()


if __name__ == "__main__":
    # 测试
    try:
        predictor = SAM3Predictor()
        print("SAM3Predictor 初始化成功！")
    except Exception as e:
        print(f"初始化失败: {e}")
