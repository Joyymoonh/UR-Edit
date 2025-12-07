#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试视频编辑 Worker 接口
演示如何使用 run_video_edit 和 get_video_edit_status
"""

import sys
import os
import time
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.video_edit_interface import VideoEditRequest, get_video_edit_status
from src.video_edit_worker import run_video_edit


def test_video_edit():
    """测试视频编辑功能"""
    
    # 创建测试请求
    job_id = f"test_{int(time.time())}"
    output_dir = Path("outputs") / job_id
    
    request = VideoEditRequest(
        job_id=job_id,
        input_video_path=Path("examples/rgb.mp4"),  # 请根据实际情况修改
        output_dir=output_dir,
        mask_prompt="hair",
        edit_prompt="make the hair red",
        max_frames=30,  # 测试用，只处理30帧
        num_inference_steps=20,  # 测试用，减少步数
        image_guidance_scale=1.5,
        guidance_scale=9.0,
        erode_kernel=10,
        apply_smoothing=True,
        smoothing_window=5,
        smoothing_alpha=0.3,
        extra={
            "model_path": os.getenv("INSTRUCT_PIX2PIX_MODEL_PATH"),
            "sam_path": os.getenv("SAM3_MODEL_PATH"),
        }
    )
    
    print("=" * 70)
    print("测试视频编辑 Worker")
    print("=" * 70)
    print(f"任务ID: {job_id}")
    print(f"输入视频: {request.input_video_path}")
    print(f"输出目录: {request.output_dir}")
    print(f"分割提示: {request.mask_prompt}")
    print(f"编辑指令: {request.edit_prompt}")
    print("=" * 70)
    
    # 检查输入文件是否存在
    if not request.input_video_path.exists():
        print(f"错误: 输入视频不存在: {request.input_video_path}")
        print("请修改脚本中的 input_video_path 或创建测试视频")
        return
    
    # 启动编辑任务（同步执行）
    print("\n启动视频编辑任务...")
    try:
        run_video_edit(request)
        print("\n✓ 任务执行完成")
    except Exception as e:
        print(f"\n✗ 任务执行失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 轮询状态（演示）
    print("\n轮询任务状态...")
    for i in range(5):
        status = get_video_edit_status(job_id, output_dir)
        print(f"\n状态查询 {i+1}:")
        print(f"  状态: {status.status}")
        print(f"  进度: {status.progress:.2%}")
        if status.current_frame is not None:
            print(f"  当前帧: {status.current_frame}/{status.total_frames}")
        if status.message:
            print(f"  消息: {status.message}")
        if status.error:
            print(f"  错误: {status.error}")
        
        if status.status in ["finished", "failed"]:
            break
        
        time.sleep(2)
    
    # 显示最终结果
    final_status = get_video_edit_status(job_id, output_dir)
    print("\n" + "=" * 70)
    print("最终状态")
    print("=" * 70)
    print(f"状态: {final_status.status}")
    print(f"进度: {final_status.progress:.2%}")
    if final_status.output_video_path:
        print(f"输出视频: {final_status.output_video_path}")
    if final_status.mask_preview_path:
        print(f"Mask预览: {final_status.mask_preview_path}")
    if final_status.logs_path:
        print(f"日志文件: {final_status.logs_path}")
    if final_status.error:
        print(f"错误信息: {final_status.error}")
    print("=" * 70)


if __name__ == "__main__":
    test_video_edit()

