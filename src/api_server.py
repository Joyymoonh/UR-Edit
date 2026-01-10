#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI ???
---------------
?? `video_edit_worker.run_video_edit` ? `get_video_edit_status`??????
HTTP ????? UI/????? REST ????????????

?????
    uvicorn src.api_server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from pathlib import Path
from typing import Dict, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.video_edit_interface import EditMode, VideoEditRequest, VideoEditStatus
from src import video_edit_worker  # noqa: F401 - ensure interface is patched
from src.video_edit_worker import get_video_edit_status, run_video_edit

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")


def _default_outputs_root() -> Path:
    return Path(os.getenv("UR_EDIT_OUTPUT_DIR", "outputs")).resolve()


class JobCreatePayload(BaseModel):
    """
    HTTP ???????? VideoEditRequest???????? job_id?output_dir?
    """

    job_id: Optional[str] = Field(default=None, description="???? job_id????????")
    input_video_path: str = Field(..., description="???????????")
    output_dir: Optional[str] = Field(default=None, description="??????? outputs/<job_id>")
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
    extra: Dict[str, object] = Field(default_factory=dict, description="??? worker ?????")

    def to_request(self, outputs_root: Path) -> VideoEditRequest:
        job_id = self.job_id or f"job_{uuid.uuid4().hex[:8]}"
        output_dir = Path(self.output_dir) if self.output_dir else outputs_root / job_id
        extra = dict(self.extra)
        extra.setdefault("job_name", extra.get("job_name", job_id))
        return VideoEditRequest(
            job_id=job_id,
            input_video_path=Path(self.input_video_path),
            output_dir=output_dir,
            mask_prompt=self.mask_prompt,
            edit_prompt=self.edit_prompt,
            mode=self.mode,
            max_frames=self.max_frames,
            image_guidance_scale=self.image_guidance_scale,
            guidance_scale=self.guidance_scale,
            erode_kernel=self.erode_kernel,
            num_inference_steps=self.num_inference_steps,
            apply_smoothing=self.apply_smoothing,
            smoothing_window=self.smoothing_window,
            smoothing_alpha=self.smoothing_alpha,
            extra=extra,
        )


class JobCreateResponse(BaseModel):
    job_id: str
    status: str
    output_dir: str


class StatusResponse(BaseModel):
    job_id: str
    status: str
    progress: float
    current_frame: Optional[int]
    total_frames: Optional[int]
    message: Optional[str]
    error: Optional[str]
    output_video_path: Optional[str]
    mask_preview_path: Optional[str]
    logs_path: Optional[str]

    @classmethod
    def from_dataclass(cls, status: VideoEditStatus) -> "StatusResponse":
        payload = status.to_dict()
        return cls(**payload)


app = FastAPI(title="UR-Edit Video Worker API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

outputs_root = _default_outputs_root()
active_threads: Dict[str, threading.Thread] = {}
threads_lock = threading.Lock()


def _run_job_task(request: VideoEditRequest) -> None:
    """???????????????????"""
    try:
        logger.info("Starting job %s", request.job_id)
        run_video_edit(request)
        logger.info("Job %s finished", request.job_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Job %s failed: %s", request.job_id, exc)
    finally:
        with threads_lock:
            active_threads.pop(request.job_id, None)


@app.post("/jobs", response_model=JobCreateResponse)
async def create_job(payload: JobCreatePayload, background_tasks: BackgroundTasks) -> JobCreateResponse:
    request = payload.to_request(outputs_root)
    if not request.input_video_path.exists():
        raise HTTPException(status_code=400, detail=f"???????: {request.input_video_path}")

    with threads_lock:
        if request.job_id in active_threads:
            raise HTTPException(status_code=409, detail=f"Job {request.job_id} ????")

        thread = threading.Thread(target=_run_job_task, args=(request,), daemon=True)
        active_threads[request.job_id] = thread
        background_tasks.add_task(thread.start)

    logger.info("Job %s queued (output_dir=%s)", request.job_id, request.output_dir)
    return JobCreateResponse(job_id=request.job_id, status="queued", output_dir=str(request.output_dir))


@app.get("/jobs/{job_id}", response_model=StatusResponse)
async def get_job_status(job_id: str) -> StatusResponse:
    try:
        status = get_video_edit_status(job_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if status.status == "failed" and status.error and "???" in status.error:
        raise HTTPException(status_code=404, detail=status.error)

    return StatusResponse.from_dataclass(status)


@app.get("/jobs/{job_id}/logs")
async def get_job_log(job_id: str):
    """????????????? log ?????????????/???"""
    status = get_video_edit_status(job_id)
    if not status.logs_path or not Path(status.logs_path).exists():
        raise HTTPException(status_code=404, detail="???????")
    return {"job_id": job_id, "logs_path": str(status.logs_path)}
