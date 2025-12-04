#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Instruct-Pix2Pix 编辑器模块
结合 SAM3 mask 进行区域编辑
(基于 LDM 原生实现，无需 diffusers 和联网)
"""

import sys
import os

# Debug: 检查 taming 是否可导入
try:
    import taming
    print(f"Debug: taming module found at {taming.__file__}")
except ImportError as e:
    print(f"Debug: Failed to import taming at startup: {e}")

import math
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageOps
import einops
from einops import rearrange
from torch import autocast
from omegaconf import OmegaConf

# 添加 instruct-pix2pix 路径以导入依赖
INSTRUCT_PATH = os.path.join(os.path.dirname(__file__), '..', 'instruct-pix2pix')
if INSTRUCT_PATH not in sys.path:
    sys.path.append(INSTRUCT_PATH)
    sys.path.append(os.path.join(INSTRUCT_PATH, "stable_diffusion"))

try:
    import k_diffusion as K
    from stable_diffusion.ldm.util import instantiate_from_config
except ImportError as e:
    print(f"Error importing LDM modules: {e}")
    print("请确保 instruct-pix2pix 子模块已正确安装并包含完整源码")
    sys.exit(1)


class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
        cfg_z = einops.repeat(z, "1 ... -> n ...", n=3)
        cfg_sigma = einops.repeat(sigma, "1 ... -> n ...", n=3)
        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], uncond["c_crossattn"][0]])],
            "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
        }
        out_cond, out_img_cond, out_uncond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
        return out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_uncond)


class InstructPix2PixEditor:
    """
    Instruct-Pix2Pix 编辑器 (LDM 版本)
    支持全图编辑和mask区域编辑
    """
    
    def __init__(self, 
                 model_id="/home/zmh/SAM3-Video-Editor/instruct-pix2pix/checkpoints/instruct-pix2pix-00-22000.ckpt", 
                 device="cuda",
                 config_path="/home/zmh/SAM3-Video-Editor/instruct-pix2pix/configs/generate.yaml"):
        """
        初始化 Instruct-Pix2Pix 模型
        
        Args:
            model_id: 模型checkpoint路径 (.ckpt)
            device: 设备
            config_path: 模型配置文件路径 (.yaml)
        """
        self.device = device
        
        print(f"加载 Instruct-Pix2Pix 模型: {model_id}")
        print(f"配置文件: {config_path}")
        
        if not os.path.exists(config_path):
            # 尝试使用相对路径
            config_path = os.path.join(INSTRUCT_PATH, "configs", "generate.yaml")
            
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件未找到: {config_path}")

        # 加载配置和模型
        config = OmegaConf.load(config_path)
        self.model = self._load_model_from_config(config, model_id)
        self.model.eval().to(device)
        
        # 初始化包装器
        self.model_wrap = K.external.CompVisDenoiser(self.model)
        self.model_wrap_cfg = CFGDenoiser(self.model_wrap)
        self.null_token = self.model.get_learned_conditioning([""])
        
        print("✓ Instruct-Pix2Pix 加载成功")
    
    def _load_model_from_config(self, config, ckpt, verbose=False):
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"]
        model = instantiate_from_config(config.model)
        m, u = model.load_state_dict(sd, strict=False)
        if len(m) > 0 and verbose:
            print("missing keys:", m)
        if len(u) > 0 and verbose:
            print("unexpected keys:", u)
        return model
    
    def edit_image(self,
                   image: Image.Image,
                   prompt: str,
                   mask: torch.Tensor = None,
                   num_inference_steps: int = 50,
                   image_guidance_scale: float = 1.5,
                   guidance_scale: float = 7.5,
                   negative_prompt: str = "", # LDM 版本通常不需要 negative prompt，或者使用空字符串
                   seed: int = None) -> Image.Image:
        """
        编辑图像
        """
        # 如果有mask，先处理mask区域
        if mask is not None:
            return self._edit_with_mask(
                image, prompt, mask,
                num_inference_steps, image_guidance_scale, guidance_scale, seed
            )
        
        # 全图编辑
        edited = self._run_inference(
            image, prompt, 
            num_inference_steps, image_guidance_scale, guidance_scale, seed
        )
        
        return edited
    
    def _run_inference(self, input_image, prompt, steps, cfg_image, cfg_text, seed):
        if seed is None:
            seed = torch.randint(0, 100000, (1,)).item()
            
        # 图像预处理
        width, height = input_image.size
        
        # 确保尺寸是64的倍数 (LDM 要求)
        new_w = math.ceil(width / 64) * 64
        new_h = math.ceil(height / 64) * 64
        if new_w != width or new_h != height:
            input_image = input_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
        with torch.no_grad(), autocast("cuda"), self.model.ema_scope():
            cond = {}
            cond["c_crossattn"] = [self.model.get_learned_conditioning([prompt])]
            
            # 图像归一化 -1 到 1
            img_tensor = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
            img_tensor = rearrange(img_tensor, "h w c -> 1 c h w").to(self.device)
            
            cond["c_concat"] = [self.model.encode_first_stage(img_tensor).mode()]

            uncond = {}
            uncond["c_crossattn"] = [self.null_token]
            uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]

            sigmas = self.model_wrap.get_sigmas(steps)

            extra_args = {
                "cond": cond,
                "uncond": uncond,
                "text_cfg_scale": cfg_text,
                "image_cfg_scale": cfg_image,
            }
            
            torch.manual_seed(seed)
            z = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
            z = K.sampling.sample_euler_ancestral(self.model_wrap_cfg, z, sigmas, extra_args=extra_args)
            
            x = self.model.decode_first_stage(z)
            x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
            x = 255.0 * rearrange(x, "1 c h w -> h w c")
            edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())
            
            # 如果调整过大小，调回去
            if new_w != width or new_h != height:
                edited_image = edited_image.resize((width, height), Image.Resampling.LANCZOS)
                
            return edited_image

    def _edit_with_mask(self,
                        image: Image.Image,
                        prompt: str,
                        mask: torch.Tensor,
                        num_inference_steps: int,
                        image_guidance_scale: float,
                        guidance_scale: float,
                        seed: int) -> Image.Image:
        
        # 1. 全图编辑
        edited = self._run_inference(
            image, prompt, 
            num_inference_steps, image_guidance_scale, guidance_scale, seed
        )
        
        # 2. 使用mask混合
        image_np = np.array(image).astype(np.float32)
        edited_np = np.array(edited).astype(np.float32)
        
        # 处理mask
        if isinstance(mask, torch.Tensor):
            mask_np = mask[0, 0].cpu().numpy()
        else:
            mask_np = mask
            
        # 调整mask大小
        if mask_np.shape != image_np.shape[:2]:
            from PIL import Image as PILImage
            mask_pil = PILImage.fromarray((mask_np * 255).astype(np.uint8))
            mask_pil = mask_pil.resize(image.size, PILImage.BILINEAR)
            mask_np = np.array(mask_pil).astype(np.float32) / 255.0
            
        mask_np = np.expand_dims(mask_np, axis=-1)
        mask_np = self._feather_mask(mask_np, feather_amount=5)
        
        result_np = mask_np * edited_np + (1 - mask_np) * image_np
        result_np = np.clip(result_np, 0, 255).astype(np.uint8)
        
        return Image.fromarray(result_np)

    def _feather_mask(self, mask: np.ndarray, feather_amount: int = 5) -> np.ndarray:
        import cv2
        if feather_amount > 0:
            kernel_size = feather_amount * 2 + 1
            mask = cv2.GaussianBlur(
                mask,
                (kernel_size, kernel_size),
                feather_amount / 2
            )
            if len(mask.shape) == 2:
                mask = np.expand_dims(mask, axis=-1)
        return mask

if __name__ == "__main__":
    print("InstructPix2PixEditor (LDM) 模块已加载")
