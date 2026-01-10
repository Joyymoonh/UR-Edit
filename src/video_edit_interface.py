#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared interface definitions for coordinating UI/backend collaboration.

This file captures the payload schemas and function declarations that both the
UI-facing code and the GPU inference workers must respect. By relying on the
same dataclasses we make sure a request produced on the UI side can be passed
directly into the worker entry points without additional translation layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Literal, Optional, Any


JobStatus = Literal[
    "queued",
    "extracting",
    "masking",
    "propagating",
    "editing",
    "smoothing",
    "saving",
    "finished",
    "failed",
]

EditMode = Literal["single_frame", "video"]


@dataclass
class VideoEditRequest:
    """
    Canonical payload used to launch a SAM3 + Instruct-Pix2Pix editing job.

    TODO(UI/前端):
        - 在 UI / 后端控制层中构造此 dataclass，填入 job_id、输入输出路径以及
          所有可调参数，再将其序列化后发送给协作者提供的接口。
        - 负责生成唯一 job_id、上传素材到服务器，并保证路径在服务器可访问。

    中文说明:
        - 该结构体描述一次视频/单帧任务所需的全部参数，UI 侧将其序列化后发送
          给服务器，服务器严格按照字段含义执行既定的 SAM3 + Instruct-Pix2Pix
          处理流程。
        - `job_id` 由 UI 生成，贯穿日志、状态和输出文件命名。
        - `input_video_path`、`output_dir` 均为服务器侧可访问的路径，目的是让
          协作者无需再猜测文件位置。
        - 其余字段正对 `scripts/edit_video_distributed.py` 与
          `SAM3InstructVideoEditor.edit_video` 的参数，默认值与项目 README 保持
          一致，保证双方行为一致。
    """

    job_id: str
    input_video_path: Path
    output_dir: Path
    mask_prompt: str
    edit_prompt: str
    mode: EditMode = "video"
    max_frames: Optional[int] = None
    image_guidance_scale: float = 1.5
    guidance_scale: float = 9.0
    erode_kernel: int = 10
    num_inference_steps: int = 50
    apply_smoothing: bool = True
    smoothing_window: int = 5
    smoothing_alpha: float = 0.3
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize into primitives for JSON/logging."""
        return {
            "job_id": self.job_id,
            "input_video_path": str(self.input_video_path),
            "output_dir": str(self.output_dir),
            "mask_prompt": self.mask_prompt,
            "edit_prompt": self.edit_prompt,
            "mode": self.mode,
            "max_frames": self.max_frames,
            "image_guidance_scale": self.image_guidance_scale,
            "guidance_scale": self.guidance_scale,
            "erode_kernel": self.erode_kernel,
            "num_inference_steps": self.num_inference_steps,
            "apply_smoothing": self.apply_smoothing,
            "smoothing_window": self.smoothing_window,
            "smoothing_alpha": self.smoothing_alpha,
            "extra": self.extra,
        }


@dataclass
class VideoEditStatus:
    """
    Shared status structure for progress polling.

    TODO(UI/前端):
        - 定时调用 `get_video_edit_status`，将返回的字段显示在 UI（状态、进度、
          错误、输出下载链接等）。
        - 当 `status` 为 finished 或 failed 时，触发结果下载、报警提示等后续操作。

    中文说明:
        - UI 轮询此结构体以展示实时进度，状态取值与 README 中“编辑阶段”定义相
          对应，覆盖提帧、分割、掩膜传播、推理、平滑、保存等节点。
        - `progress` 建议代表整体 0~1 百分比；若无法精确统计至少更新阶段枚举。
        - `output_video_path`、`mask_preview_path`、`logs_path` 用于 UI 展示结果
          和调试信息，路径统一返回服务器端绝对路径或可下载 URL。
    """

    job_id: str
    status: JobStatus
    progress: float = 0.0
    current_frame: Optional[int] = None
    total_frames: Optional[int] = None
    message: Optional[str] = None
    error: Optional[str] = None
    output_video_path: Optional[Path] = None
    mask_preview_path: Optional[Path] = None
    logs_path: Optional[Path] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for transport."""
        return {
            "job_id": self.job_id,
            "status": self.status,
            "progress": self.progress,
            "current_frame": self.current_frame,
            "total_frames": self.total_frames,
            "message": self.message,
            "error": self.error,
            "output_video_path": str(self.output_video_path) if self.output_video_path else None,
            "mask_preview_path": str(self.mask_preview_path) if self.mask_preview_path else None,
            "logs_path": str(self.logs_path) if self.logs_path else None,
        }


def run_video_edit(request: VideoEditRequest) -> None:
    """
    Entry point that GPU 侧协作者必须实现.

    TODO(协作者):
        - 将此函数对接到 GPU 推理脚本，支持 VideoEditRequest 中的全部字段。
        - 按阶段更新 VideoEditStatus 并写入 request.output_dir/status.json。

    中文说明:
        - GPU 端负责实现该函数，将 `request` 转换成真实的执行流程，可直接调
          用 `SAM3InstructVideoEditor` 或 `scripts/edit_video_distributed.py`。
        - 建议以 `request.output_dir` 作为唯一的工作目录：保存日志、mask
          可视化、最终视频以及 `status.json`；这样 UI 仅凭 `job_id` 就能获
          取完整产物。
        - 函数可以同步执行（阻塞至完成）也可以将任务投递到后台队列，但无论哪
          种实现方式都必须在开始执行前写入一条“queued”状态，并在结束时更新为
          “finished/failed”。
    """
    raise NotImplementedError("GPU worker must implement run_video_edit")


def get_video_edit_status(job_id: str, output_dir: Optional[Path] = None) -> VideoEditStatus:
    """
    Status polling hook used by UI/REST 层.

    TODO(协作者):
        - 读取最新 status.json（或数据库记录）并返回 VideoEditStatus。
        - 当 job 不存在时抛出异常或返回 failed 状态 + error。

    中文说明:
        - UI 或上层服务通过该函数了解任务执行阶段，推荐实现方式是读取
          `output_dir/status.json` 并反序列化成 `VideoEditStatus`。
        - 若任务不存在或已被清理，应抛出明确异常或返回 `status="failed"` 并
          写入 `error` 字段，避免无响应。
        - `output_dir` 参数可选，如果不提供，实现应该尝试从常见位置查找。
    
    Args:
        job_id: 任务ID
        output_dir: 可选的输出目录路径
    """
    
    raise NotImplementedError("GPU worker must implement get_video_edit_status")
