#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试单帧编辑效果 (Instruct-Pix2Pix)
对比全图编辑与 Crop-Edit-Paste 局部编辑的效果
"""

import argparse
import torch
import cv2
import os
import numpy as np
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

# 导入 SAM3 单图预测器
import sys
# 将项目根目录加入路径 (假设脚本在 scripts/ 下)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.sam3_masking import SAM3Predictor

def main():
    parser = argparse.ArgumentParser(description="测试单帧编辑效果")
    parser.add_argument("--input", "-i", default="/home/zmh/SAM3-Video-Editor/b64b82ca-afb4-4894-856f-6dd1a7618492.png", help="输入视频路径")
    parser.add_argument("--output", "-o", default="outputs/test_frame0.png", help="输出图片路径")
    parser.add_argument("--mask-prompt", "-m", default="luckin coffee sign", help="分割目标")
    parser.add_argument("--edit-prompt", "-e", default="turn it into a starbucks sign", help="编辑指令")
    parser.add_argument("--steps", type=int, default=50, help="推理步数")
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="文本引导")
    parser.add_argument("--image-guidance-scale", type=float, default=1.5, help="图像引导")
    parser.add_argument("--erode-kernel", type=int, default=0, help="Mask腐蚀大小，用于收缩边缘保护背景")
    parser.add_argument("--model-path", default="/home/zmh/SAM3-Video-Editor/instruct-pix2pix/diffusers_model", help="转换后的模型路径")
    parser.add_argument("--sam-path", default="/home/zmh/SAM3-Video-Editor/pretrained_models/sam3/sam3.pt", help="SAM3模型路径")
    args = parser.parse_args()

    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("=" * 60)
    print("  单帧编辑效果测试 (Frame 0)")
    print("=" * 60)
    
    # 1. 读取输入
    print(f"读取输入: {args.input}")
    if args.input.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        original_image = Image.open(args.input).convert("RGB")
    else:
        # 假设是视频
        cap = cv2.VideoCapture(args.input)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("无法读取视频/图片")
            return
        original_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    w, h = original_image.size
    print(f"原图尺寸: {w}x{h}")
    
    # 2. 加载 SAM3 并生成 Mask
    print(f"\n加载 SAM3: '{args.mask_prompt}'")
    sam = SAM3Predictor(checkpoint_path=args.sam_path, device="cuda:0")
    mask_tensor = sam.generate_mask(original_image, args.mask_prompt)
    mask_np = mask_tensor.cpu().numpy()[0, 0] # (H, W)
    
    # Mask 腐蚀 (Shrink)
    if args.erode_kernel > 0:
        print(f"正在腐蚀 Mask (Kernel Size: {args.erode_kernel})...")
        kernel = np.ones((args.erode_kernel, args.erode_kernel), np.uint8)
        # 注意: mask_np 是 bool 或 float，需要转 uint8 处理后再转回
        mask_uint8 = (mask_np > 0).astype(np.uint8)
        mask_eroded = cv2.erode(mask_uint8, kernel, iterations=1)
        mask_np = mask_eroded.astype(bool)
    
    # 释放 SAM3
    del sam
    torch.cuda.empty_cache()
    
    if mask_np.max() == 0:
        print("警告: 未检测到目标物体，将跳过 Crop 测试")
        has_mask = False
    else:
        has_mask = True
        # 保存 Mask 预览
        mask_vis = Image.fromarray((mask_np * 255).astype(np.uint8))
        mask_vis.save("outputs/test_frame0_mask.png")
        print("Mask 已保存到 outputs/test_frame0_mask.png")

    # 3. 加载 Instruct-Pix2Pix
    print(f"\n加载 Instruct-Pix2Pix...")
    pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        local_files_only=True,
        safety_checker=None,
        device_map="balanced"
    )
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)

    # 4. 测试 A: 全图编辑 (Resize 模式)
    print("\n[Test A] 全图编辑 (Resize to 512)...")
    # 缩放长边到 512
    scale = 512 / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    new_w = new_w - (new_w % 8)
    new_h = new_h - (new_h % 8)
    
    input_resized = original_image.resize((new_w, new_h), Image.LANCZOS)
    
    with torch.no_grad():
        edited_a = pipeline(
            args.edit_prompt,
            image=input_resized,
            num_inference_steps=args.steps,
            image_guidance_scale=args.image_guidance_scale,
            guidance_scale=args.guidance_scale
        ).images[0]
        
    edited_a_full = edited_a.resize((w, h), Image.LANCZOS)
    edited_a_full.save("outputs/test_frame0_full_edit.png")
    print("全图编辑结果已保存: outputs/test_frame0_full_edit.png")

    # 5. 测试 B: Crop-Edit-Paste (如果 mask 存在)
    if has_mask:
        print("\n[Test B] Crop-Edit-Paste 局部编辑...")
        
        # 计算 BBox
        rows, cols = np.where(mask_np > 0)
        y_min, y_max = rows.min(), rows.max()
        x_min, x_max = cols.min(), cols.max()
        
        # Padding
        pad = 50
        y_min = max(0, y_min - pad)
        y_max = min(h, y_max + pad)
        x_min = max(0, x_min - pad)
        x_max = min(w, x_max + pad)
        
        crop_box = (x_min, y_min, x_max, y_max)
        crop_img = original_image.crop(crop_box)
        cw, ch = crop_img.size
        print(f"Crop 区域: {crop_box}, 尺寸: {cw}x{ch}")
        
        # Resize Crop to 512 (for better quality)
        scale_crop = 512 / max(cw, ch)
        cw_new = int(cw * scale_crop)
        ch_new = int(ch * scale_crop)
        cw_new = cw_new - (cw_new % 8)
        ch_new = ch_new - (ch_new % 8)
        
        crop_input = crop_img.resize((cw_new, ch_new), Image.LANCZOS)
        
        with torch.no_grad():
            edited_crop = pipeline(
                args.edit_prompt,
                image=crop_input,
                num_inference_steps=args.steps,
                image_guidance_scale=args.image_guidance_scale,
                guidance_scale=args.guidance_scale
            ).images[0]
            
        # Resize back
        edited_crop_orig = edited_crop.resize((cw, ch), Image.LANCZOS)
        
        # --- 关键修正：只贴回 Mask 区域，防止背景穿帮 ---
        # 1. 获取该 Crop 区域对应的原始 Mask
        crop_mask_np = mask_np[y_min:y_max, x_min:x_max]
        crop_mask_img = Image.fromarray((crop_mask_np * 255).astype(np.uint8))
        
        # 2. 稍微羽化 Mask 边缘，让融合更自然
        from PIL import ImageFilter
        crop_mask_img = crop_mask_img.resize((cw, ch), Image.NEAREST) # 确保尺寸一致
        crop_mask_blurred = crop_mask_img.filter(ImageFilter.GaussianBlur(radius=3))
        
        # 3. Paste with Mask
        result_b = original_image.copy()
        result_b.paste(edited_crop_orig, crop_box, mask=crop_mask_blurred)
        # -----------------------------------------------
        
        result_b.save("outputs/test_frame0_crop_edit.png")
        print("Crop 编辑结果已保存: outputs/test_frame0_crop_edit.png")
        
        # 保存 Crop 本身的对比
        crop_img.save("outputs/test_frame0_crop_input.png")
        edited_crop_orig.save("outputs/test_frame0_crop_output.png")
        crop_mask_blurred.save("outputs/test_frame0_crop_mask.png") # 保存一下融合用的 mask 供检查

    print("\n测试完成！")

if __name__ == "__main__":
    main()
