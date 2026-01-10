#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分布式高清视频编辑器核心逻辑
从 edit_video_distributed.py 提取，支持状态回调
"""

import torch
import cv2
import os
import numpy as np
import shutil
import sys
from pathlib import Path
from PIL import Image, ImageFilter
from tqdm import tqdm
from typing import Optional, Callable, List, TextIO
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

# 导入 SAM3
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sam3.sam3.model.sam3_video_predictor import Sam3VideoPredictor

from src.video_edit_interface import JobStatus


def save_frames_to_temp(frames: List[Image.Image], temp_dir: str = "temp_frames") -> str:
    """保存帧到临时目录供 SAM3 Video Predictor 使用"""
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    print(f"保存帧到临时目录: {temp_dir}")
    for i, frame in enumerate(tqdm(frames, desc="Saving frames")):
        frame.save(os.path.join(temp_dir, f"{i:05d}.jpg"), quality=100)
    return temp_dir


def process_video_edit(
    input_video_path: Path,
    output_video_path: Path,
    mask_prompt: str,
    edit_prompt: str,
    max_frames: Optional[int] = None,
    num_inference_steps: int = 50,
    image_guidance_scale: float = 1.5,
    guidance_scale: float = 9.0,
    erode_kernel: int = 10,
    apply_smoothing: bool = True,
    smoothing_window: int = 5,
    smoothing_alpha: float = 0.3,
    status_callback: Optional[Callable] = None,
    log_file: Optional[TextIO] = None,
    model_path: Optional[str] = None,
    sam_path: Optional[str] = None,
):
    """
    核心视频编辑处理函数
    
    Args:
        input_video_path: 输入视频路径
        output_video_path: 输出视频路径
        mask_prompt: SAM3分割提示词
        edit_prompt: 编辑指令
        max_frames: 最大处理帧数
        num_inference_steps: 推理步数
        image_guidance_scale: 图像引导强度
        guidance_scale: 文本引导强度
        erode_kernel: Mask腐蚀核大小
        apply_smoothing: 是否应用时序平滑
        smoothing_window: 平滑窗口大小
        smoothing_alpha: 平滑强度
        status_callback: 状态更新回调函数
        log_file: 日志文件对象
        model_path: Instruct-Pix2Pix模型路径
        sam_path: SAM3模型路径
    """
    def log_print(*args, **kwargs):
        """同时输出到控制台和日志文件"""
        print(*args, **kwargs)
        if log_file:
            print(*args, **kwargs, file=log_file)
            log_file.flush()
    
    # 默认路径
    if model_path is None:
        model_path = os.getenv("INSTRUCT_PIX2PIX_MODEL_PATH", 
                              "/home/zmh/SAM3-Video-Editor/instruct-pix2pix/diffusers_model")
    if sam_path is None:
        sam_path = os.getenv("SAM3_MODEL_PATH",
                            "/home/zmh/SAM3-Video-Editor/pretrained_models/sam3/sam3.pt")
    
    # 1. 初始化 Diffusers 模型
    if status_callback:
        status_callback("extracting", 0.0, message="加载 Instruct-Pix2Pix 模型...")
    
    log_print("=" * 60)
    log_print("  分布式高清视频编辑器 v2.0 (Crop-Edit-Paste + Mask Fusion)")
    log_print("=" * 60)
    log_print(f"加载 Instruct-Pix2Pix (from {model_path})...")
    log_print("启用 device_map='balanced' (利用多GPU)...")
    
    pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        local_files_only=True,
        safety_checker=None,
        device_map="balanced"
    )
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
    log_print("✓ Diffusers 模型加载成功 (Model Parallel Enabled)")
    
    # 2. 提取视频帧
    if status_callback:
        status_callback("extracting", 0.1, message="提取视频帧...")
    
    log_print(f"\n提取视频: {input_video_path}")
    cap = cv2.VideoCapture(str(input_video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    log_print(f"分辨率: {width}x{height} (全分辨率，无缩放)")
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    total_frames = len(frames)
    log_print(f"总帧数: {total_frames}")
    
    if status_callback:
        status_callback("extracting", 0.2, total_frames=total_frames, message=f"已提取 {total_frames} 帧")
    
    # 3. SAM3 视频跟踪
    if status_callback:
        status_callback("masking", 0.25, message="初始化 SAM3 Video Predictor...")
    
    log_print(f"\n[阶段 1] 初始化 SAM3 Video Predictor 进行纯文本跟踪")
    
    # 保存帧到临时目录
    temp_frames_dir = save_frames_to_temp(frames, temp_dir=f"temp_frames_{os.getpid()}")
    
    try:
        # 初始化 SAM3 Video Predictor
        sam_predictor = Sam3VideoPredictor(
            checkpoint_path=sam_path,
            video_loader_type="cv2"
        )
        
        # 启动会话
        log_print("启动 SAM3 会话...")
        session = sam_predictor.start_session(resource_path=temp_frames_dir)
        session_id = session["session_id"]
        
        # 直接使用文本提示初始化
        log_print(f"使用文本提示进行初始化: '{mask_prompt}' @ Frame 0")
        sam_predictor.add_prompt(
            session_id=session_id,
            frame_idx=0,
            text=mask_prompt,
            obj_id=1
        )
        
        # 视频传播 (Tracking)
        if status_callback:
            status_callback("propagating", 0.3, current_frame=0, total_frames=total_frames, 
                          message="执行 SAM3 全视频跟踪...")
        
        log_print("执行 SAM3 全视频跟踪...")
        masks = [None] * total_frames
        
        for output in tqdm(sam_predictor.propagate_in_video(
            session_id=session_id,
            start_frame_idx=0,
            propagation_direction="forward",
            max_frame_num_to_track=total_frames
        ), total=total_frames, desc="SAM3 Tracking"):
            
            frame_idx = output["frame_index"]
            
            # 更新状态
            if status_callback:
                progress = 0.3 + 0.2 * (frame_idx + 1) / total_frames
                status_callback("propagating", progress, current_frame=frame_idx + 1, 
                              total_frames=total_frames, message=f"跟踪进度: {frame_idx + 1}/{total_frames}")
            
            # output["outputs"] 包含 "pred_masks"
            if "pred_masks" in output["outputs"]:
                pred_masks = output["outputs"]["pred_masks"]
                if isinstance(pred_masks, torch.Tensor):
                    mask = pred_masks[0].cpu().numpy()
                else:
                    mask = pred_masks[0]
                    
                # 确保 mask 是 (H, W)
                if mask.ndim == 3:
                    mask = mask[0]
                    
                masks[frame_idx] = (mask > 0).astype(np.float32)
            else:
                masks[frame_idx] = np.zeros((height, width), dtype=np.float32)
        
        log_print("✓ SAM3 跟踪完成")
        
        # 关闭会话并释放显存
        sam_predictor.close_session(session_id)
        del sam_predictor
        torch.cuda.empty_cache()
        
    except Exception as e:
        log_print(f"SAM3 跟踪失败: {e}")
        import traceback
        log_print(traceback.format_exc())
        raise
    finally:
        if os.path.exists(temp_frames_dir):
            shutil.rmtree(temp_frames_dir)
    
    # 4. 编辑 (High Res + Crop-Edit-Paste + Mask Fusion)
    if status_callback:
        status_callback("editing", 0.5, current_frame=0, total_frames=total_frames,
                      message="开始高清编辑 (Crop-Edit-Paste w/ Mask Fusion)...")
    
    log_print(f"\n开始高清编辑 (Crop-Edit-Paste w/ Mask Fusion): '{edit_prompt}'")
    edited_frames = []
    
    # 准备 Erosion Kernel
    if erode_kernel > 0:
        kernel = np.ones((erode_kernel, erode_kernel), np.uint8)
        log_print(f"启用 Mask 腐蚀，Kernel Size: {erode_kernel}")
    
    for i, (frame, mask) in enumerate(tqdm(zip(frames, masks), total=total_frames, desc="Editing")):
        # 更新状态
        if status_callback:
            progress = 0.5 + 0.4 * (i + 1) / total_frames
            status_callback("editing", progress, current_frame=i + 1, total_frames=total_frames,
                          message=f"编辑进度: {i + 1}/{total_frames}")
        
        if mask is None or mask.sum() == 0:
            edited_frames.append(frame)
            continue
            
        # --- Mask 预处理 (腐蚀) ---
        if erode_kernel > 0:
            mask_uint8 = (mask > 0).astype(np.uint8)
            mask_eroded = cv2.erode(mask_uint8, kernel, iterations=1)
            mask = mask_eroded.astype(np.float32)
            
        if mask.sum() == 0:  # 腐蚀完没了
            edited_frames.append(frame)
            continue
            
        # 1. 计算 BBox
        rows, cols = np.where(mask > 0)
        y_min, y_max = rows.min(), rows.max()
        x_min, x_max = cols.min(), cols.max()
        
        # Padding
        pad = 50
        h_img, w_img = frame.size[1], frame.size[0]  # PIL size is (W, H)
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
                edit_prompt,
                image=crop_input,
                num_inference_steps=num_inference_steps,
                image_guidance_scale=image_guidance_scale,
                guidance_scale=guidance_scale,
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
    
    # 5. 时序平滑（如果需要）
    if apply_smoothing:
        if status_callback:
            status_callback("smoothing", 0.9, total_frames=total_frames, message="应用时序平滑...")
        
        log_print("\n应用时序平滑...")
        frames_np = [np.array(f).astype(np.float32) for f in edited_frames]
        smoothed_frames = []
        for i in tqdm(range(len(frames_np)), desc="平滑"):
            start = max(0, i - smoothing_window // 2)
            end = min(len(frames_np), i + smoothing_window // 2 + 1)
            window = frames_np[start:end]
            center_idx = i - start
            weights = np.exp(-np.abs(np.arange(len(window)) - center_idx) / 2.0)
            weights /= weights.sum()
            smoothed = np.average(window, axis=0, weights=weights)
            smoothed = smoothing_alpha * smoothed + (1 - smoothing_alpha) * frames_np[i]
            smoothed = np.clip(smoothed, 0, 255).astype(np.uint8)
            smoothed_frames.append(Image.fromarray(smoothed))
        edited_frames = smoothed_frames
    
    # 6. 保存视频
    if status_callback:
        status_callback("saving", 0.95, total_frames=total_frames, message="保存视频...")
    
    log_print(f"\n保存视频: {output_video_path}")
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    
    # 同时保存 Mask 视频
    mask_output_path = output_video_path.with_suffix("").with_name(
        output_video_path.stem + "_mask.mp4"
    )
    log_print(f"保存 Mask 预览: {mask_output_path}")
    out_mask = cv2.VideoWriter(str(mask_output_path), fourcc, fps, (width, height))
    
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
    log_print("✓ 完成！")

