# 分布式视频编辑 Worker 实现总结

## 已完成的工作

### ✅ 1. 接口实现 (`src/video_edit_worker.py`)

实现了 `video_edit_interface.py` 中定义的两个核心接口：

- **`run_video_edit(request: VideoEditRequest) -> None`**
  - 接收 `VideoEditRequest` 对象
  - 创建输出目录并初始化状态为 `queued`
  - 调用核心处理函数执行视频编辑
  - 实时更新状态文件 `status.json`
  - 处理完成后更新为 `finished` 或 `failed`

- **`get_video_edit_status(job_id: str, output_dir: Optional[Path] = None) -> VideoEditStatus`**
  - 从 `status.json` 读取任务状态
  - 支持自动查找输出目录
  - 任务不存在时返回 `failed` 状态

### ✅ 2. 核心处理逻辑重构 (`scripts/edit_video_distributed_core.py`)

将 `edit_video_distributed.py` 的核心逻辑提取为独立函数：

- **`process_video_edit(...)`** - 核心处理函数
  - 支持状态回调（`status_callback`）
  - 支持日志文件输出（`log_file`）
  - 完整的视频编辑流程：
    1. 加载 Instruct-Pix2Pix 模型
    2. 提取视频帧
    3. SAM3 视频跟踪
    4. Crop-Edit-Paste 编辑
    5. 时序平滑（可选）
    6. 保存视频和 Mask 预览

### ✅ 3. 状态管理机制

- 状态文件：`output_dir/status.json`
- 状态流转：
  ```
  queued → extracting → masking → propagating → editing → smoothing → saving → finished
                                                                    ↓
                                                                  failed
  ```
- 状态字段包含：
  - `status`: 当前阶段
  - `progress`: 进度百分比 (0.0 ~ 1.0)
  - `current_frame` / `total_frames`: 帧进度
  - `message`: 状态消息
  - `error`: 错误信息（如果失败）
  - `output_video_path`: 输出视频路径
  - `mask_preview_path`: Mask 预览路径
  - `logs_path`: 日志文件路径

### ✅ 4. 输出文件规范

所有输出文件保存在 `output_dir` 目录：

- `{job_id}_output.mp4` - 编辑后的视频
- `{job_id}_mask.mp4` - Mask 预览视频
- `{job_id}_log.txt` - 处理日志
- `status.json` - 状态文件

### ✅ 5. 向后兼容

- `scripts/edit_video_distributed.py` 保持向后兼容
- 可以继续通过命令行使用
- 如果核心模块可用，会自动使用新实现

## 文件结构

```
UR-Edit/
├── src/
│   ├── video_edit_interface.py          # 接口定义（已更新）
│   └── video_edit_worker.py              # Worker 实现（新增）
├── scripts/
│   ├── edit_video_distributed.py        # 命令行脚本（已更新，向后兼容）
│   ├── edit_video_distributed_core.py  # 核心处理逻辑（新增）
│   └── test_video_edit_worker.py        # 测试脚本（新增）
└── docs/
    ├── API_USAGE.md                     # API 使用文档（新增）
    └── WORKER_IMPLEMENTATION.md         # 本文档（新增）
```

## 使用方式

### 方式 1: 通过接口调用（推荐，用于 UI 集成）

```python
from pathlib import Path
from src.video_edit_interface import VideoEditRequest
from src.video_edit_worker import run_video_edit, get_video_edit_status

# 创建请求
request = VideoEditRequest(
    job_id="test_001",
    input_video_path=Path("input.mp4"),
    output_dir=Path("outputs/test_001"),
    mask_prompt="hair",
    edit_prompt="make the hair red",
)

# 启动任务
run_video_edit(request)

# 查询状态
status = get_video_edit_status("test_001", Path("outputs/test_001"))
print(f"状态: {status.status}, 进度: {status.progress:.2%}")
```

### 方式 2: 命令行使用（向后兼容）

```bash
python scripts/edit_video_distributed.py \
  --input input.mp4 \
  --output output.mp4 \
  --mask-prompt "hair" \
  --edit-prompt "make the hair red" \
  --max-frames 60 \
  --erode-kernel 10
```

## UI 集成指南

### 1. 启动任务

```python
from src.video_edit_interface import VideoEditRequest
from src.video_edit_worker import run_video_edit
import threading

# 创建请求（从 UI 表单获取参数）
request = VideoEditRequest(
    job_id=generate_unique_job_id(),  # UI 生成唯一ID
    input_video_path=Path(uploaded_video_path),  # 上传后的路径
    output_dir=Path(f"outputs/{job_id}"),
    mask_prompt=user_mask_prompt,
    edit_prompt=user_edit_prompt,
    **user_params  # 其他参数
)

# 异步执行
thread = threading.Thread(target=run_video_edit, args=(request,))
thread.start()
```

### 2. 轮询状态

```python
from src.video_edit_worker import get_video_edit_status
import time

def poll_status(job_id, output_dir):
    while True:
        status = get_video_edit_status(job_id, output_dir)
        
        # 更新 UI
        update_ui_progress(status.progress)
        update_ui_status(status.status)
        update_ui_message(status.message)
        
        if status.status in ["finished", "failed"]:
            break
        
        time.sleep(2)  # 每2秒轮询一次
```

### 3. 处理结果

```python
status = get_video_edit_status(job_id, output_dir)

if status.status == "finished":
    # 下载视频
    download_file(status.output_video_path)
    # 显示预览
    show_preview(status.mask_preview_path)
elif status.status == "failed":
    # 显示错误
    show_error(status.error)
    # 显示日志
    show_logs(status.logs_path)
```

## 测试

运行测试脚本：

```bash
# 确保设置了必要的环境变量
export INSTRUCT_PIX2PIX_MODEL_PATH="/path/to/diffusers_model"
export SAM3_MODEL_PATH="/path/to/sam3.pt"

# 运行测试
python scripts/test_video_edit_worker.py
```

## 注意事项

1. **路径要求**：
   - 所有路径必须是服务器可访问的绝对路径
   - `output_dir` 会自动创建，确保有写入权限

2. **模型路径**：
   - 可以通过环境变量设置：`INSTRUCT_PIX2PIX_MODEL_PATH`、`SAM3_MODEL_PATH`
   - 也可以在 `request.extra` 中指定

3. **状态文件**：
   - 状态文件实时更新，UI 可以安全地轮询
   - 任务完成后状态文件会保留，直到手动清理

4. **错误处理**：
   - 所有异常都会被捕获并写入状态文件
   - 日志文件包含详细的错误堆栈

5. **性能**：
   - 支持多GPU并行（自动分配）
   - 使用 Crop-Edit-Paste 策略，只编辑目标区域
   - 支持时序平滑，减少闪烁

## 下一步工作

对于 UI 开发者：

1. ✅ **接口已实现** - 可以直接调用 `run_video_edit` 和 `get_video_edit_status`
2. ⏳ **UI 集成** - 需要实现：
   - 任务提交界面
   - 状态轮询机制
   - 进度显示
   - 结果下载

3. ⏳ **可选优化**：
   - 任务队列管理（Celery/Redis）
   - 异步任务执行
   - 结果缓存
   - 用户认证

## 相关文档

- [API 使用说明](API_USAGE.md) - 详细的 API 使用文档
- [README.md](../README.md) - 项目总体说明
- [video_edit_interface.py](../src/video_edit_interface.py) - 接口定义

