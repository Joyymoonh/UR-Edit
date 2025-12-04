#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分布式高清视频编辑器 (8卡 3090 专用版)
使用 Diffusers + Accelerate 实现模型并行，支持全分辨率编辑
集成 SAM3 Video Tracking 实现全视频智能跟踪
增强：支持 Mask 腐蚀与局部 Mask 融合 (Crop-Edit-Paste with Mask)
"""

import argparse
import torch
import cv2
import os
import numpy as np
import shutil
import glob
import sys
from PIL import Image, ImageFilter
from tqdm import tqdm
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from accelerate import Accelerator

# 导入 SAM3
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 注意：这里直接导入 sam3 的 predictor
from sam3.sam3.model.sam3_video_predictor import Sam3VideoPredictor

def save_frames_to_temp(frames, temp_dir="temp_frames"):
    """保存帧到临时目录供 SAM3 Video Predictor 使用"""
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    print(f"保存帧到临时目录: {temp_dir}")
    for i, frame in enumerate(tqdm(frames, desc="Saving frames")):
        frame.save(os.path.join(temp_dir, f"{i:05d}.jpg"), quality=100)
    return temp_dir

def main():
    parser = argparse.ArgumentParser(description="分布式高清视频编辑器")
    parser.add_argument("--input", "-i", required=True, help="输入视频路径")
    parser.add_argument("--output", "-o", required=True, help="输出视频路径")
    parser.add_argument("--mask-prompt", "-m", required=True, help="分割目标")
    parser.add_argument("--edit-prompt", "-e", required=True, help="编辑指令")
    parser.add_argument("--steps", type=int, default=50, help="推理步数")
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="文本引导")
    parser.add_argument("--image-guidance-scale", type=float, default=1.5, help="图像引导")
    parser.add_argument("--model-path", default="/home/zmh/SAM3-Video-Editor/instruct-pix2pix/diffusers_model", help="转换后的模型路径")
    parser.add_argument("--sam-path", default="/home/zmh/SAM3-Video-Editor/pretrained_models/sam3/sam3.pt", help="SAM3模型路径")
    parser.add_argument("--max-frames", type=int, default=None, help="最大处理帧数")
    parser.add_argument("--erode-kernel", type=int, default=0, help="Mask腐蚀大小，用于收缩边缘保护背景 (推荐 5-15)")
    args = parser.parse_args()

    print("=" * 60)
    print("  分布式高清视频编辑器 v2.0 (Crop-Edit-Paste + Mask Fusion)")
    print("=" * 60)
    
    # 1. 初始化 Diffusers 模型 (自动分配到多卡)
    print(f"加载 Instruct-Pix2Pix (from {args.model_path})...")
    print("启用 device_map='balanced' (利用 8x3090)...")
    
    pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        local_files_only=True,
        safety_checker=None,
        device_map="balanced"
    )
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
    print("✓ Diffusers 模型加载成功 (Model Parallel Enabled)")

    # 2. 处理视频帧
    print(f"\n提取视频: {args.input}")
    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"分辨率: {width}x{height} (全分辨率，无缩放)")
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        if args.max_frames and len(frames) >= args.max_frames:
            break
    cap.release()
    print(f"总帧数: {len(frames)}")
    
    # 3. SAM3 视频跟踪 (Running on GPU0)
    print(f"\n[阶段 1] 初始化 SAM3 Video Predictor 进行纯文本跟踪")
    
    # 保存帧到临时目录
    temp_frames_dir = save_frames_to_temp(frames)
    
    try:
        # 初始化 SAM3 Video Predictor
        sam_predictor = Sam3VideoPredictor(
            checkpoint_path=args.sam_path,
            video_loader_type="cv2" # 从目录加载
        )
        
        # 启动会话
        print("启动 SAM3 会话...")
        session = sam_predictor.start_session(resource_path=temp_frames_dir)
        session_id = session["session_id"]
        
        # 直接使用文本提示初始化
        print(f"使用文本提示进行初始化: '{args.mask_prompt}' @ Frame 0")
        sam_predictor.add_prompt(
            session_id=session_id,
            frame_idx=0,
            text=args.mask_prompt,
            obj_id=1
        )
        
        # 视频传播 (Tracking)
        print("执行 SAM3 全视频跟踪...")
        masks = [None] * len(frames)
        
        for output in tqdm(sam_predictor.propagate_in_video(
            session_id=session_id,
            start_frame_idx=0,
            propagation_direction="forward", # 从0开始向后
            max_frame_num_to_track=len(frames)
        ), total=len(frames), desc="SAM3 Tracking"):
            
            frame_idx = output["frame_index"]
            
            # output["outputs"] 包含 "pred_masks"
            if "pred_masks" in output["outputs"]:
                pred_masks = output["outputs"]["pred_masks"]
                if isinstance(pred_masks, torch.Tensor):
                    mask = pred_masks[0].cpu().numpy() # 取第一个对象的 mask
                else:
                    mask = pred_masks[0]
                    
                # 确保 mask 是 (H, W)
                if mask.ndim == 3:
                    mask = mask[0] 
                    
                masks[frame_idx] = (mask > 0).astype(np.float32)
            else:
                masks[frame_idx] = np.zeros((height, width), dtype=np.float32)

        print("✓ SAM3 跟踪完成")
        
        # 关闭会话并释放显存
        sam_predictor.close_session(session_id)
        del sam_predictor
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"SAM3 跟踪失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if os.path.exists(temp_frames_dir):
            shutil.rmtree(temp_frames_dir)

    # 5. 编辑 (High Res + Crop-Edit-Paste + Mask Fusion)
    print(f"\n开始高清编辑 (Crop-Edit-Paste w/ Mask Fusion): '{args.edit_prompt}'")
    edited_frames = []
    
    # 准备 Erosion Kernel
    if args.erode_kernel > 0:
        kernel = np.ones((args.erode_kernel, args.erode_kernel), np.uint8)
        print(f"启用 Mask 腐蚀，Kernel Size: {args.erode_kernel}")
    
    for i, (frame, mask) in enumerate(tqdm(zip(frames, masks), total=len(frames), desc="Editing")):
        if mask is None or mask.sum() == 0:
            edited_frames.append(frame)
            continue
            
        # --- Mask 预处理 (腐蚀) ---
        if args.erode_kernel > 0:
            mask_uint8 = (mask > 0).astype(np.uint8)
            mask_eroded = cv2.erode(mask_uint8, kernel, iterations=1)
            mask = mask_eroded.astype(np.float32)
            
        if mask.sum() == 0: # 腐蚀完没了
            edited_frames.append(frame)
            continue
            
        # 1. 计算 BBox
        rows, cols = np.where(mask > 0)
        y_min, y_max = rows.min(), rows.max()
        x_min, x_max = cols.min(), cols.max()
        
        # Padding
        pad = 50
        h_img, w_img = frame.size[1], frame.size[0] # PIL size is (W, H)
        y_min = max(0, y_min - pad)
        y_max = min(h_img, y_max + pad)
        x_min = max(0, x_min - pad)
        x_max = min(w_img, x_max + pad)
        
        # 2. Crop
        crop_box = (x_min, y_min, x_max, y_max)
        crop_img = frame.crop(crop_box)
        
        # 3. Resize to 512
        cw, ch = crop_img.size
        scale_factor = 512 / max(cw, ch)
        new_w = int(cw * scale_factor)
        new_h = int(ch * scale_factor)
        new_w = new_w - (new_w % 8)
        new_h = new_h - (new_h % 8)
        
        crop_input = crop_img.resize((new_w, new_h), Image.LANCZOS)
            
        # 4. Instruct-Pix2Pix Inference
        with torch.no_grad():
            edited_crop = pipeline(
                args.edit_prompt,
                image=crop_input,
                num_inference_steps=args.steps,
                image_guidance_scale=args.image_guidance_scale,
                guidance_scale=args.guidance_scale,
                output_type="pil"
            ).images[0]
            
        # 5. 恢复尺寸
        edited_crop_orig_size = edited_crop.resize((cw, ch), Image.LANCZOS)
        
        # 6. --- Mask Fusion (关键步骤) ---
        # 获取该 Crop 区域的 Mask
        crop_mask_np = mask[y_min:y_max, x_min:x_max]
        crop_mask_img = Image.fromarray((crop_mask_np * 255).astype(np.uint8))
        
        # Resize mask to match crop (in case of rounding errors)
        crop_mask_img = crop_mask_img.resize((cw, ch), Image.NEAREST)
        
        # 羽化 Mask
        crop_mask_blurred = crop_mask_img.filter(ImageFilter.GaussianBlur(radius=5))
        
        # Paste with Mask
        final_frame = frame.copy()
        final_frame.paste(edited_crop_orig_size, crop_box, mask=crop_mask_blurred)
        
        edited_frames.append(final_frame)

    # 6. 保存视频
    print(f"\n保存视频: {args.output}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    # 同时保存 Mask 视频
    mask_output_path = args.output.replace(".mp4", "_mask.mp4")
    print(f"保存 Mask 预览: {mask_output_path}")
    out_mask = cv2.VideoWriter(mask_output_path, fourcc, fps, (width, height))
    
    for i, f in enumerate(edited_frames):
        out.write(cv2.cvtColor(np.array(f), cv2.COLOR_RGB2BGR))
        
        if i < len(masks) and masks[i] is not None:
            mask_vis = (masks[i] * 255).astype(np.uint8)
            mask_vis = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)
            out_mask.write(mask_vis)
        else:
            out_mask.write(np.zeros((height, width, 3), dtype=np.uint8))
            
    out.release()
    out_mask.release()
    print("✓ 完成！")

if __name__ == "__main__":
    main()
