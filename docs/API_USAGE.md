# 视频编辑 Worker API 使用说明

本文档说明如何使用 `video_edit_interface.py` 中定义的接口进行视频编辑。

## 概述

项目已经实现了符合 UI 接口规范的分布式视频编辑功能：

- ✅ **单帧测试** (`scripts/single_frame_edit.py`) - 已完成
- ✅ **分布式视频编辑 Worker** (`src/video_edit_worker.py`) - 已完成
- ✅ **核心处理逻辑** (`scripts/edit_video_distributed_core.py`) - 已完成

## 接口说明

### 1. `run_video_edit(request: VideoEditRequest) -> None`

启动视频编辑任务。

**参数说明：**

```python
from src.video_edit_interface import VideoEditRequest

request = VideoEditRequest(
    job_id="unique_job_id",              # 唯一任务ID（由UI生成）
    input_video_path=Path("input.mp4"), # 输入视频路径
    output_dir=Path("outputs/job_id"),  # 输出目录
    mask_prompt="hair",                  # SAM3分割提示词
    edit_prompt="make the hair red",     # 编辑指令
    mode="video",                        # 模式: "single_frame" 或 "video"
    max_frames=None,                    # 最大处理帧数（None=全部）
    image_guidance_scale=1.5,           # 图像引导强度
    guidance_scale=9.0,                 # 文本引导强度
    erode_kernel=10,                    # Mask腐蚀核大小
    num_inference_steps=50,             # 推理步数
    apply_smoothing=True,               # 是否应用时序平滑
    smoothing_window=5,                 # 平滑窗口大小
    smoothing_alpha=0.3,                # 平滑强度
    extra={                             # 额外参数
        "model_path": "...",            # Instruct-Pix2Pix模型路径（可选）
        "sam_path": "..."               # SAM3模型路径（可选）
    }
)
```

**工作流程：**

1. 函数会在 `output_dir` 创建 `status.json` 文件
2. 状态会依次更新：`queued` → `extracting` → `masking` → `propagating` → `editing` → `smoothing` → `saving` → `finished`
3. 输出文件保存在 `output_dir`：
   - `{job_id}_output.mp4` - 编辑后的视频
   - `{job_id}_mask.mp4` - Mask预览视频
   - `{job_id}_log.txt` - 处理日志
   - `status.json` - 状态文件

### 2. `get_video_edit_status(job_id: str, output_dir: Optional[Path] = None) -> VideoEditStatus`

查询任务状态。

**返回值：**

```python
from src.video_edit_interface import VideoEditStatus

status = get_video_edit_status("job_id", output_dir=Path("outputs/job_id"))

# status 包含以下字段：
# - job_id: str
# - status: JobStatus ("queued" | "extracting" | "masking" | "propagating" | 
#                      "editing" | "smoothing" | "saving" | "finished" | "failed")
# - progress: float (0.0 ~ 1.0)
# - current_frame: Optional[int]
# - total_frames: Optional[int]
# - message: Optional[str]
# - error: Optional[str]
# - output_video_path: Optional[Path]
# - mask_preview_path: Optional[Path]
# - logs_path: Optional[Path]
```

## 使用示例

### 示例 1: 基本使用

```python
from pathlib import Path
from src.video_edit_interface import VideoEditRequest
from src.video_edit_worker import run_video_edit, get_video_edit_status
import time

# 创建请求
job_id = "test_001"
request = VideoEditRequest(
    job_id=job_id,
    input_video_path=Path("examples/rgb.mp4"),
    output_dir=Path("outputs") / job_id,
    mask_prompt="hair",
    edit_prompt="make the hair red",
    max_frames=60,
)

# 启动任务（同步执行）
try:
    run_video_edit(request)
    print("任务完成")
except Exception as e:
    print(f"任务失败: {e}")

# 查询状态
status = get_video_edit_status(job_id, request.output_dir)
print(f"状态: {status.status}, 进度: {status.progress:.2%}")
```

### 示例 2: 异步执行 + 状态轮询

```python
import threading
from pathlib import Path
from src.video_edit_interface import VideoEditRequest
from src.video_edit_worker import run_video_edit, get_video_edit_status
import time

def run_task(request):
    """在后台线程中运行任务"""
    try:
        run_video_edit(request)
    except Exception as e:
        print(f"任务失败: {e}")

# 创建请求
job_id = "test_002"
request = VideoEditRequest(
    job_id=job_id,
    input_video_path=Path("examples/rgb.mp4"),
    output_dir=Path("outputs") / job_id,
    mask_prompt="person",
    edit_prompt="make the person wear a red shirt",
)

# 在后台线程启动任务
thread = threading.Thread(target=run_task, args=(request,))
thread.start()

# 轮询状态
while thread.is_alive():
    status = get_video_edit_status(job_id, request.output_dir)
    print(f"[{status.status}] 进度: {status.progress:.2%} | {status.message or ''}")
    
    if status.status in ["finished", "failed"]:
        break
    
    time.sleep(2)

# 显示最终结果
final_status = get_video_edit_status(job_id, request.output_dir)
if final_status.status == "finished":
    print(f"✓ 完成！输出: {final_status.output_video_path}")
else:
    print(f"✗ 失败: {final_status.error}")
```

### 示例 3: REST API 集成（伪代码）

```python
from flask import Flask, request, jsonify
from pathlib import Path
from src.video_edit_interface import VideoEditRequest
from src.video_edit_worker import run_video_edit, get_video_edit_status
import threading

app = Flask(__name__)

@app.route("/api/video/edit", methods=["POST"])
def start_edit():
    """启动编辑任务"""
    data = request.json
    
    job_id = data["job_id"]
    request_obj = VideoEditRequest(
        job_id=job_id,
        input_video_path=Path(data["input_video_path"]),
        output_dir=Path(data["output_dir"]),
        mask_prompt=data["mask_prompt"],
        edit_prompt=data["edit_prompt"],
        **data.get("params", {})
    )
    
    # 异步执行
    thread = threading.Thread(target=run_video_edit, args=(request_obj,))
    thread.start()
    
    return jsonify({"job_id": job_id, "status": "queued"})

@app.route("/api/video/status/<job_id>", methods=["GET"])
def get_status(job_id):
    """查询任务状态"""
    output_dir = request.args.get("output_dir")
    status = get_video_edit_status(job_id, Path(output_dir) if output_dir else None)
    return jsonify(status.to_dict())
```

## 环境变量配置

可以通过环境变量指定模型路径（可选）：

```bash
export INSTRUCT_PIX2PIX_MODEL_PATH="/path/to/diffusers_model"
export SAM3_MODEL_PATH="/path/to/sam3.pt"
export UR_EDIT_OUTPUT_DIR="outputs"  # 默认输出目录
```

如果不设置，代码会使用默认路径。

## 注意事项

1. **路径要求**：
   - `input_video_path` 必须是服务器可访问的绝对路径
   - `output_dir` 会自动创建，确保有写入权限

2. **状态文件**：
   - 状态文件保存在 `output_dir/status.json`
   - UI 可以通过轮询该文件获取实时状态

3. **错误处理**：
   - 如果任务失败，状态会更新为 `failed`，错误信息在 `error` 字段
   - 日志文件会记录详细的错误堆栈

4. **性能优化**：
   - 支持多GPU并行（通过 `device_map="balanced"`）
   - 使用 Crop-Edit-Paste 策略，只编辑目标区域
   - 支持时序平滑，减少闪烁

## 测试

运行测试脚本：

```bash
python scripts/test_video_edit_worker.py
```

确保：
- 输入视频路径正确
- 模型路径已配置（或使用环境变量）
- 有足够的GPU显存

## 与 UI 集成

UI 开发者需要：

1. **启动任务**：
   ```python
   from src.video_edit_interface import VideoEditRequest
   from src.video_edit_worker import run_video_edit
   
   request = VideoEditRequest(...)
   run_video_edit(request)  # 可以异步调用
   ```

2. **轮询状态**：
   ```python
   from src.video_edit_worker import get_video_edit_status
   
   status = get_video_edit_status(job_id, output_dir)
   # 更新UI显示
   ```

3. **处理结果**：
   ```python
   if status.status == "finished":
       # 下载 output_video_path
       # 显示 mask_preview_path
   elif status.status == "failed":
       # 显示错误信息
   ```

## 相关文件

- `src/video_edit_interface.py` - 接口定义
- `src/video_edit_worker.py` - Worker 实现
- `scripts/edit_video_distributed_core.py` - 核心处理逻辑
- `scripts/test_video_edit_worker.py` - 测试脚本

