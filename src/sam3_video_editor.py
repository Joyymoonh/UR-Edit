#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAM3 + Instruct-Pix2Pix 视频编辑器
结合目标追踪和文本指导编辑
"""

import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import gc
from typing import Optional, Tuple, List

from .sam3_masking import SAM3Predictor
from .instruct_pix2pix_editor import InstructPix2PixEditor


class SAM3InstructVideoEditor:
    """
    SAM3 + Instruct-Pix2Pix 视频编辑器
    
    工作流程:
    1. 使用SAM3分割第一帧的目标
    2. 使用光流追踪mask到所有帧
    3. 使用Instruct-Pix2Pix编辑mask区域
    4. 应用时序平滑
    5. 合成视频
    """
    
    def __init__(self,
                 sam_checkpoint: Optional[str] = "/home/zmh/SAM3-Video-Editor/pretrained_models/sam3/sam3.pt",
                 instruct_pix2pix_model: str = "/home/zmh/SAM3-Video-Editor/instruct-pix2pix/checkpoints/instruct-pix2pix-00-22000.ckpt",
                 device: str = "cuda"):
        """
        初始化编辑器
        """
        self.device = device
        self.sam_checkpoint = sam_checkpoint
        self.instruct_pix2pix_model = instruct_pix2pix_model
        
        # 模型实例（懒加载）
        self.sam = None
        self.editor = None
        
        print("=" * 70)
        print("初始化 SAM3 + Instruct-Pix2Pix 视频编辑器")
        print("=" * 70)
    
    def load_sam(self):
        """加载SAM3模型"""
        if self.sam is None:
            print("\n[1/2] 加载 SAM3...")
            self.sam = SAM3Predictor(checkpoint_path=self.sam_checkpoint, device=self.device)
            print("SAM3 loaded successfully!")

    def unload_sam(self):
        """卸载SAM3模型以释放显存"""
        if self.sam is not None:
            print("\n卸载 SAM3 以释放显存...")
            del self.sam
            self.sam = None
            gc.collect()
            torch.cuda.empty_cache()
            print("✓ SAM3 已卸载")

    def load_editor(self):
        """加载Instruct-Pix2Pix模型"""
        if self.editor is None:
            print("\n[2/2] 加载 Instruct-Pix2Pix...")
            self.editor = InstructPix2PixEditor(model_id=self.instruct_pix2pix_model, device=self.device)
            print("✓ Instruct-Pix2Pix 加载成功")

    def unload_editor(self):
        """卸载Instruct-Pix2Pix模型"""
        if self.editor is not None:
            print("\n卸载 Instruct-Pix2Pix...")
            del self.editor
            self.editor = None
            gc.collect()
            torch.cuda.empty_cache()
            print("✓ Instruct-Pix2Pix 已卸载")

    def extract_frames(self,
                      video_path: str,
                      max_frames: Optional[int] = None,
                      max_size: Optional[int] = 512) -> Tuple[List[Image.Image], int, Tuple[int, int]]:
        """
        提取视频帧并调整大小
        
        Args:
            video_path: 视频路径
            max_frames: 最大帧数
            max_size: 最大长边尺寸，用于避免OOM。建议 512-1024。
        """
        print(f"\n提取视频帧: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        print(f"  原始分辨率: {width}x{height}, FPS: {fps}, 总帧数: {total_frames}")
        
        # 计算新的尺寸
        new_width, new_height = width, height
        if max_size and (width > max_size or height > max_size):
            scale = max_size / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            print(f"  调整后分辨率: {new_width}x{new_height} (scale={scale:.2f})")
        
        frames = []
        pbar = tqdm(total=total_frames, desc="提取帧")
        
        while len(frames) < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # BGR → RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            
            # Resize if needed
            if new_width != width or new_height != height:
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
            frames.append(img)
            pbar.update(1)
        
        cap.release()
        pbar.close()
        
        print(f"✓ 提取了 {len(frames)} 帧")
        
        return frames, fps, (new_width, new_height)
    
    def segment_first_frame(self,
                           first_frame: Image.Image,
                           mask_prompt: str) -> torch.Tensor:
        """
        使用SAM3分割第一帧
        """
        self.load_sam() # 确保加载
        
        print(f"\n使用SAM3分割第一帧: '{mask_prompt}'")
        mask = self.sam.generate_mask(first_frame, mask_prompt)
        print(f"✓ 分割完成，mask shape: {mask.shape}")
        return mask
    
    def compute_optical_flow(self,
                            frame1: Image.Image,
                            frame2: Image.Image) -> np.ndarray:
        """计算光流"""
        # 转换为灰度图
        frame1_gray = cv2.cvtColor(np.array(frame1), cv2.COLOR_RGB2GRAY)
        frame2_gray = cv2.cvtColor(np.array(frame2), cv2.COLOR_RGB2GRAY)
        
        # 计算光流
        flow = cv2.calcOpticalFlowFarneback(
            frame1_gray, frame2_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        
        return flow
    
    def warp_mask(self, mask: torch.Tensor, flow: np.ndarray) -> torch.Tensor:
        """Warp mask"""
        mask_np = mask[0, 0].cpu().numpy()
        h, w = mask_np.shape
        
        flow_x, flow_y = flow[:, :, 0], flow[:, :, 1]
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        map_x = (x + flow_x).astype(np.float32)
        map_y = (y + flow_y).astype(np.float32)
        
        warped_mask_np = cv2.remap(
            mask_np,
            map_x, map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        warped_mask_np = (warped_mask_np > 0.5).astype(np.float32)
        warped_mask = torch.from_numpy(warped_mask_np).unsqueeze(0).unsqueeze(0)
        
        return warped_mask.to(self.device)
    
    def propagate_masks(self,
                       frames: List[Image.Image],
                       first_mask: torch.Tensor) -> List[torch.Tensor]:
        """使用光流传播mask"""
        print(f"\n传播mask到 {len(frames)} 帧...")
        masks = [first_mask]
        for i in tqdm(range(1, len(frames)), desc="传播mask"):
            flow = self.compute_optical_flow(frames[i-1], frames[i])
            mask_i = self.warp_mask(masks[-1], flow)
            masks.append(mask_i)
        print(f"✓ Mask传播完成")
        return masks
    
    def edit_frames(self,
                   frames: List[Image.Image],
                   masks: List[torch.Tensor],
                   edit_prompt: str,
                   num_inference_steps: int = 50,
                   image_guidance_scale: float = 1.5,
                   guidance_scale: float = 7.5) -> List[Image.Image]:
        """使用Instruct-Pix2Pix编辑所有帧"""
        self.load_editor() # 确保加载
        
        print(f"\n使用Instruct-Pix2Pix编辑 {len(frames)} 帧...")
        print(f"  编辑指令: '{edit_prompt}'")
        print(f"  推理步数: {num_inference_steps}")
        
        edited_frames = []
        
        # 启用 half precision 以节省显存
        with torch.cuda.amp.autocast():
            for i, (frame, mask) in enumerate(tqdm(zip(frames, masks), total=len(frames), desc="编辑帧")):
                edited = self.editor.edit_image(
                    image=frame,
                    prompt=edit_prompt,
                    mask=mask,
                    num_inference_steps=num_inference_steps,
                    image_guidance_scale=image_guidance_scale,
                    guidance_scale=guidance_scale
                )
                edited_frames.append(edited)
                
                # 每一帧后清理一下
                if i % 10 == 0:
                    torch.cuda.empty_cache()
        
        print(f"✓ 编辑完成")
        return edited_frames
    
    def temporal_smoothing(self, frames: List[Image.Image], window_size: int = 5, alpha: float = 0.3) -> List[Image.Image]:
        """时序平滑"""
        print(f"\n应用时序平滑...")
        frames_np = [np.array(f).astype(np.float32) for f in frames]
        smoothed_frames = []
        for i in tqdm(range(len(frames_np)), desc="平滑"):
            start = max(0, i - window_size // 2)
            end = min(len(frames_np), i + window_size // 2 + 1)
            window = frames_np[start:end]
            center_idx = i - start
            weights = np.exp(-np.abs(np.arange(len(window)) - center_idx) / 2.0)
            weights /= weights.sum()
            smoothed = np.average(window, axis=0, weights=weights)
            smoothed = alpha * smoothed + (1 - alpha) * frames_np[i]
            smoothed = np.clip(smoothed, 0, 255).astype(np.uint8)
            smoothed_frames.append(Image.fromarray(smoothed))
        return smoothed_frames
    
    def save_video(self, frames: List[Image.Image], output_path: str, fps: int = 30):
        """保存视频"""
        print(f"\n保存视频: {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        height, width = np.array(frames[0]).shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        for frame in tqdm(frames, desc="写入视频"):
            frame_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        out.release()
        print(f"✓ 视频已保存: {output_path}")
    
    def _save_mask_visualization(self, frame: Image.Image, mask: torch.Tensor, output_path: str):
        """保存mask可视化"""
        import matplotlib.pyplot as plt
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(frame)
        axes[0].set_title("Original Frame")
        axes[0].axis('off')
        
        mask_np = mask[0, 0].cpu().numpy()
        axes[1].imshow(mask_np, cmap='gray')
        axes[1].set_title("SAM3 Mask")
        axes[1].axis('off')
        
        frame_np = np.array(frame)
        overlay = frame_np.copy()
        overlay[mask_np > 0.5] = overlay[mask_np > 0.5] * 0.5 + np.array([255, 0, 0]) * 0.5
        axes[2].imshow(overlay.astype(np.uint8))
        axes[2].set_title("Overlay")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  保存mask可视化: {output_path}")

    def edit_video(self,
                  video_path: str,
                  mask_prompt: str,
                  edit_prompt: str,
                  output_path: str,
                  max_frames: Optional[int] = None,
                  num_inference_steps: int = 50,
                  apply_smoothing: bool = True,
                  smoothing_window: int = 5,
                  smoothing_alpha: float = 0.3,
                  max_size: int = 512): # 新增参数
        """完整的视频编辑流程"""
        print("\n" + "=" * 70)
        print("SAM3 + Instruct-Pix2Pix 视频编辑 (Optimized)")
        print("=" * 70)
        print(f"输入: {video_path}")
        print(f"输出: {output_path}")
        print(f"分割目标: '{mask_prompt}'")
        print(f"编辑指令: '{edit_prompt}'")
        print("=" * 70)
        
        # 1. 提取帧 (带resize)
        frames, fps, size = self.extract_frames(video_path, max_frames, max_size=max_size)
        
        # 2. 分割第一帧 (会加载SAM3)
        first_mask = self.segment_first_frame(frames[0], mask_prompt)
        
        mask_vis_path = output_path.replace('.mp4', '_first_mask.png')
        self._save_mask_visualization(frames[0], first_mask, mask_vis_path)
        
        # 3. 传播mask (此时仍然需要SAM3吗？不需要。但如果是用SAM3的VideoPropagator就需要。我们这里用的是光流，所以可以释放SAM3)
        # 注意：如果你以后想用SAM3自带的视频追踪，这里就不能unload。
        # 但目前用的是 self.propagate_masks (光流)，所以可以释放。
        self.unload_sam()
        
        masks = self.propagate_masks(frames, first_mask)
        
        # 4. 编辑帧 (会加载InstructPix2Pix)
        edited_frames = self.edit_frames(
            frames, masks, edit_prompt,
            num_inference_steps=num_inference_steps
        )
        self.unload_editor() # 释放
        
        # 5. 时序平滑
        if apply_smoothing:
            edited_frames = self.temporal_smoothing(
                edited_frames,
                window_size=smoothing_window,
                alpha=smoothing_alpha
            )
        
        # 6. 保存视频
        self.save_video(edited_frames, output_path, fps)
        
        print("\n" + "=" * 70)
        print("✓ 视频编辑完成！")
        print("=" * 70)

if __name__ == "__main__":
    print("SAM3InstructVideoEditor 模块已加载")
