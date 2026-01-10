#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频编辑 Worker 实现
实现 video_edit_interface.py 中定义的接口，对接分布式视频编辑功能
"""

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Optional, Callable
import threading

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.video_edit_interface import (
    VideoEditRequest,
    VideoEditStatus,
    JobStatus,
    run_video_edit as interface_run_video_edit,
    get_video_edit_status as interface_get_video_edit_status,
)
from scripts.edit_video_distributed_core import process_video_edit


def _update_status(status: VideoEditStatus, output_dir: Path) -> None:
    """
    更新状态文件
    
    Args:
        status: 状态对象
        output_dir: 输出目录
    """
    status_file = output_dir / "status.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(status_file, "w", encoding="utf-8") as f:
        json.dump(status.to_dict(), f, indent=2, ensure_ascii=False)


def _read_status(job_id: str, output_dir: Path) -> Optional[VideoEditStatus]:
    """
    读取状态文件
    
    Args:
        job_id: 任务ID
        output_dir: 输出目录
        
    Returns:
        状态对象，如果文件不存在则返回None
    """
    status_file = output_dir / "status.json"
    
    if not status_file.exists():
        return None
    
    try:
        with open(status_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # 重建Path对象
        if data.get("output_video_path"):
            data["output_video_path"] = Path(data["output_video_path"])
        if data.get("mask_preview_path"):
            data["mask_preview_path"] = Path(data["mask_preview_path"])
        if data.get("logs_path"):
            data["logs_path"] = Path(data["logs_path"])
        
        return VideoEditStatus(**data)
    except Exception as e:
        print(f"读取状态文件失败: {e}")
        return None


def _create_status_callback(
    request: VideoEditRequest,
    output_dir: Path
) -> Callable[[JobStatus, float, Optional[int], Optional[int], Optional[str], Optional[str]], None]:
    """
    创建状态更新回调函数
    
    Returns:
        状态更新回调函数
    """
    def update_callback(
        status: JobStatus,
        progress: float = 0.0,
        current_frame: Optional[int] = None,
        total_frames: Optional[int] = None,
        message: Optional[str] = None,
        error: Optional[str] = None
    ):
        """状态更新回调"""
        status_obj = VideoEditStatus(
            job_id=request.job_id,
            status=status,
            progress=progress,
            current_frame=current_frame,
            total_frames=total_frames,
            message=message,
            error=error,
            output_video_path=output_dir / f"{request.job_id}_output.mp4" if status == "finished" else None,
            mask_preview_path=output_dir / f"{request.job_id}_mask.mp4" if status in ["finished", "saving"] else None,
            logs_path=output_dir / f"{request.job_id}_log.txt" if status in ["finished", "failed"] else None,
        )
        _update_status(status_obj, output_dir)
    
    return update_callback


def run_video_edit(request: VideoEditRequest) -> None:
    """
    实现 video_edit_interface.py 中定义的 run_video_edit 接口
    
    该函数负责：
    1. 创建输出目录
    2. 初始化状态为 "queued"
    3. 调用核心处理函数
    4. 更新状态文件
    """
    output_dir = Path(request.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化状态为 "queued"
    initial_status = VideoEditStatus(
        job_id=request.job_id,
        status="queued",
        progress=0.0,
        message="任务已加入队列，等待处理..."
    )
    _update_status(initial_status, output_dir)
    
    # 创建状态更新回调
    status_callback = _create_status_callback(request, output_dir)
    
    try:
        # 确定输出路径
        output_video_path = output_dir / f"{request.job_id}_output.mp4"
        mask_preview_path = output_dir / f"{request.job_id}_mask.mp4"
        log_path = output_dir / f"{request.job_id}_log.txt"
        
        # 重定向日志到文件
        log_file = open(log_path, "w", encoding="utf-8")
        
        # 调用核心处理函数
        process_video_edit(
            input_video_path=Path(request.input_video_path),
            output_video_path=output_video_path,
            mask_prompt=request.mask_prompt,
            edit_prompt=request.edit_prompt,
            max_frames=request.max_frames,
            num_inference_steps=request.num_inference_steps,
            image_guidance_scale=request.image_guidance_scale,
            guidance_scale=request.guidance_scale,
            erode_kernel=request.erode_kernel,
            apply_smoothing=request.apply_smoothing,
            smoothing_window=request.smoothing_window,
            smoothing_alpha=request.smoothing_alpha,
            status_callback=status_callback,
            log_file=log_file,
            model_path=request.extra.get("model_path"),
            sam_path=request.extra.get("sam_path"),
        )
        
        log_file.close()
        
        # 更新最终状态
        final_status = VideoEditStatus(
            job_id=request.job_id,
            status="finished",
            progress=1.0,
            message="视频编辑完成",
            output_video_path=output_video_path,
            mask_preview_path=mask_preview_path,
            logs_path=log_path,
        )
        _update_status(final_status, output_dir)
        
    except Exception as e:
        error_msg = f"处理失败: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        
        # 更新失败状态
        failed_status = VideoEditStatus(
            job_id=request.job_id,
            status="failed",
            progress=0.0,
            error=error_msg,
            logs_path=output_dir / f"{request.job_id}_log.txt" if "log_file" in locals() else None,
        )
        _update_status(failed_status, output_dir)
        
        # 确保日志文件关闭
        if "log_file" in locals():
            log_file.close()
        
        raise


def get_video_edit_status(job_id: str, output_dir: Optional[Path] = None) -> VideoEditStatus:
    """
    实现 video_edit_interface.py 中定义的 get_video_edit_status 接口
    
    该函数从 status.json 读取状态，如果任务不存在则返回 failed 状态
    
    Args:
        job_id: 任务ID
        output_dir: 可选的输出目录路径。如果不提供，将尝试从常见位置查找
    
    Returns:
        VideoEditStatus 对象
    """
    # 如果提供了 output_dir，直接使用
    if output_dir is not None:
        status = _read_status(job_id, Path(output_dir))
        if status is not None:
            return status
    
    # 尝试从常见位置查找状态文件
    # 方案1: 从环境变量获取基础输出目录
    base_output_dir = os.getenv("UR_EDIT_OUTPUT_DIR", "outputs")
    base_output_path = Path(base_output_dir)
    
    # 方案2: 遍历可能的输出目录查找
    possible_dirs = [
        base_output_path / job_id,
        base_output_path,
        Path("outputs") / job_id,
        Path("outputs"),
    ]
    
    for output_dir_candidate in possible_dirs:
        status = _read_status(job_id, output_dir_candidate)
        if status is not None:
            return status
    
    # 如果找不到，返回 failed 状态
    return VideoEditStatus(
        job_id=job_id,
        status="failed",
        error=f"任务 {job_id} 不存在或已被清理。请检查输出目录是否正确。"
    )


# 覆盖接口中的占位函数
import src.video_edit_interface as interface_module
interface_module.run_video_edit = run_video_edit
interface_module.get_video_edit_status = get_video_edit_status


if __name__ == "__main__":
    # 测试代码
    print("视频编辑 Worker 模块已加载")
    print("接口函数已实现:")
    print("  - run_video_edit(request: VideoEditRequest)")
    print("  - get_video_edit_status(job_id: str) -> VideoEditStatus")

