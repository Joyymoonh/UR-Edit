# 视频编辑功能调试指南

本文档说明如何在没有UI界面的情况下调试视频编辑功能。

## 方式一：使用命令行脚本（推荐，最简单）

### 1. 准备环境

确保已安装所有依赖并配置好模型路径：

```bash
# 激活conda环境
conda activate sam3

# 设置模型路径（可选，如果不设置会使用默认路径）
export INSTRUCT_PIX2PIX_MODEL_PATH="/path/to/instruct-pix2pix/diffusers_model"
export SAM3_MODEL_PATH="/path/to/pretrained_models/sam3/sam3.pt"
```

### 2. 运行分布式视频编辑脚本

**基本命令：**

```bash
python scripts/edit_video_distributed.py \
  --input examples/rgb.mp4 \
  --output outputs/result.mp4 \
  --mask-prompt "hair" \
  --edit-prompt "make the hair red"
```

**完整参数示例：**

```bash
python scripts/edit_video_distributed.py \
  --input examples/rgb.mp4 \
  --output outputs/result.mp4 \
  --mask-prompt "hair" \
  --edit-prompt "make the hair red" \
  --max-frames 60 \
  --steps 50 \
  --guidance-scale 9.0 \
  --image-guidance-scale 1.5 \
  --erode-kernel 10 \
  --model-path /path/to/instruct-pix2pix/diffusers_model \
  --sam-path /path/to/pretrained_models/sam3/sam3.pt
```

**参数说明：**

| 参数 | 说明 | 默认值 | 推荐值 |
|------|------|--------|--------|
| `--input, -i` | 输入视频路径 | 必需 | - |
| `--output, -o` | 输出视频路径 | 必需 | - |
| `--mask-prompt, -m` | SAM3分割提示词 | 必需 | "hair", "person", "face" |
| `--edit-prompt, -e` | 编辑指令 | 必需 | "make the hair red" |
| `--max-frames` | 最大处理帧数（测试用） | None | 30-60（测试） |
| `--steps` | 推理步数 | 50 | 20-50（测试），50-100（正式） |
| `--guidance-scale` | 文本引导强度 | 7.5 | 7.5-12.0 |
| `--image-guidance-scale` | 图像引导强度 | 1.5 | 1.2-1.5（改色），0.8-1.1（结构改变） |
| `--erode-kernel` | Mask腐蚀核大小 | 0 | 5-15（推荐10） |
| `--model-path` | Instruct-Pix2Pix模型路径 | 默认路径 | - |
| `--sam-path` | SAM3模型路径 | 默认路径 | - |

### 3. 查看输出

脚本会输出：
- 编辑后的视频：`outputs/result.mp4`
- Mask预览视频：`outputs/result_mask.mp4`

---

## 方式二：使用单帧测试脚本（快速调试Prompt）

在编辑完整视频前，建议先用单帧测试效果：

```bash
python scripts/single_frame_edit.py \
  --input examples/rgb.mp4 \
  --output outputs/test_frame0.png \
  --mask-prompt "hair" \
  --edit-prompt "make the hair red" \
  --image-guidance-scale 1.5 \
  --guidance-scale 9.0 \
  --erode-kernel 15
```

**输出文件：**
- `outputs/test_frame0.png` - 编辑后的图片
- `outputs/test_frame0_mask.png` - Mask预览
- `outputs/test_frame0_full_edit.png` - 全图编辑对比
- `outputs/test_frame0_crop_edit.png` - Crop-Edit-Paste结果

---

## 方式三：使用Worker接口（测试接口功能）

如果想测试Worker接口是否正常工作：

### 1. 使用测试脚本

```bash
# 修改 scripts/test_video_edit_worker.py 中的路径
# 然后运行：
python scripts/test_video_edit_worker.py
```

### 2. 手动调用接口

创建测试脚本 `test_manual.py`：

```python
#!/usr/bin/env python3
import sys
from pathlib import Path

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent))

from src.video_edit_interface import VideoEditRequest
from src.video_edit_worker import run_video_edit, get_video_edit_status
import time

# 创建请求
job_id = "debug_test_001"
request = VideoEditRequest(
    job_id=job_id,
    input_video_path=Path("examples/rgb.mp4"),  # 修改为你的视频路径
    output_dir=Path("outputs") / job_id,
    mask_prompt="hair",
    edit_prompt="make the hair red",
    max_frames=30,  # 测试用，只处理30帧
    num_inference_steps=20,  # 测试用，减少步数
    erode_kernel=10,
)

print("=" * 70)
print("开始视频编辑任务")
print("=" * 70)

# 启动任务
try:
    run_video_edit(request)
    print("\n✓ 任务完成")
except Exception as e:
    print(f"\n✗ 任务失败: {e}")
    import traceback
    traceback.print_exc()

# 查询最终状态
status = get_video_edit_status(job_id, request.output_dir)
print(f"\n最终状态: {status.status}")
if status.output_video_path:
    print(f"输出视频: {status.output_video_path}")
if status.error:
    print(f"错误: {status.error}")
```

运行：

```bash
python test_manual.py
```

---

## 方式四：使用Python交互式调试

在Python交互环境中逐步调试：

```python
# 1. 导入模块
from pathlib import Path
from src.video_edit_interface import VideoEditRequest
from src.video_edit_worker import run_video_edit, get_video_edit_status

# 2. 创建请求
request = VideoEditRequest(
    job_id="debug_interactive",
    input_video_path=Path("examples/rgb.mp4"),
    output_dir=Path("outputs/debug_interactive"),
    mask_prompt="hair",
    edit_prompt="make the hair red",
    max_frames=10,  # 只处理10帧，快速测试
    num_inference_steps=20,
)

# 3. 执行编辑
run_video_edit(request)

# 4. 检查状态
status = get_video_edit_status("debug_interactive", Path("outputs/debug_interactive"))
print(status.to_dict())
```

---

## 常见调试场景

### 场景1：测试不同的Prompt组合

```bash
# 测试1：改发色
python scripts/edit_video_distributed.py \
  --input examples/rgb.mp4 \
  --output outputs/test1.mp4 \
  --mask-prompt "hair" \
  --edit-prompt "make the hair red" \
  --max-frames 30

# 测试2：改发型
python scripts/edit_video_distributed.py \
  --input examples/rgb.mp4 \
  --output outputs/test2.mp4 \
  --mask-prompt "hair" \
  --edit-prompt "make the hair curly and black" \
  --max-frames 30

# 测试3：改服装
python scripts/edit_video_distributed.py \
  --input examples/rgb.mp4 \
  --output outputs/test3.mp4 \
  --mask-prompt "person" \
  --edit-prompt "make the person wear a red shirt" \
  --max-frames 30
```

### 场景2：调整参数

```bash
# 测试不同的image_guidance_scale
for scale in 0.8 1.0 1.2 1.5 2.0; do
  python scripts/edit_video_distributed.py \
    --input examples/rgb.mp4 \
    --output outputs/test_scale_${scale}.mp4 \
    --mask-prompt "hair" \
    --edit-prompt "make the hair red" \
    --image-guidance-scale $scale \
    --max-frames 30
done
```

### 场景3：调试Mask问题

如果Mask覆盖了不该覆盖的区域：

```bash
# 增加erode_kernel，让Mask收缩
python scripts/edit_video_distributed.py \
  --input examples/rgb.mp4 \
  --output outputs/test_erode_15.mp4 \
  --mask-prompt "hair" \
  --edit-prompt "make the hair red" \
  --erode-kernel 15 \
  --max-frames 30
```

### 场景4：检查状态文件

```bash
# 运行任务后，查看状态文件
cat outputs/job_id/status.json

# 或者使用Python
python -c "
import json
from pathlib import Path
status = json.load(open('outputs/job_id/status.json'))
print(json.dumps(status, indent=2, ensure_ascii=False))
"
```

---

## 调试技巧

### 1. 快速测试（减少帧数和步数）

```bash
python scripts/edit_video_distributed.py \
  --input examples/rgb.mp4 \
  --output outputs/quick_test.mp4 \
  --mask-prompt "hair" \
  --edit-prompt "make the hair red" \
  --max-frames 10 \
  --steps 20
```

### 2. 查看日志

如果使用Worker接口，日志会保存在：
```
outputs/job_id/job_id_log.txt
```

### 3. 检查Mask预览

每次运行都会生成Mask预览视频：
```
outputs/result_mask.mp4
```

查看这个视频可以确认Mask是否正确。

### 4. 单帧调试

先用单帧测试，确认效果后再处理完整视频：

```bash
# 1. 单帧测试
python scripts/single_frame_edit.py \
  --input examples/rgb.mp4 \
  --mask-prompt "hair" \
  --edit-prompt "make the hair red"

# 2. 查看输出，确认效果
# 3. 如果满意，再运行完整视频
python scripts/edit_video_distributed.py \
  --input examples/rgb.mp4 \
  --output outputs/final.mp4 \
  --mask-prompt "hair" \
  --edit-prompt "make the hair red"
```

---

## 常见问题排查

### 问题1：ImportError

```
ImportError: cannot import name 'process_video_edit' from 'scripts.edit_video_distributed_core'
```

**解决：** 检查 `scripts/edit_video_distributed_core.py` 是否存在，如果不存在，脚本会自动使用向后兼容的旧代码。

### 问题2：模型路径错误

```
FileNotFoundError: 模型文件不存在
```

**解决：** 
- 检查模型路径是否正确
- 使用 `--model-path` 和 `--sam-path` 参数指定路径
- 或设置环境变量

### 问题3：OOM（显存不足）

**解决：**
- 减少 `--max-frames`（只处理部分帧）
- 减少 `--steps`（降低推理步数）
- 使用更小的视频分辨率

### 问题4：Mask效果不好

**解决：**
- 调整 `--mask-prompt`，使用更精确的描述
- 增加 `--erode-kernel`，让Mask收缩
- 查看 `*_mask.mp4` 预览视频，检查Mask是否正确

---

## 推荐调试流程

1. **单帧测试** → 确认Prompt和参数
2. **小帧数测试** → 确认流程正常（`--max-frames 10-30`）
3. **完整视频** → 处理完整视频

```bash
# 步骤1：单帧测试
python scripts/single_frame_edit.py \
  --input examples/rgb.mp4 \
  --mask-prompt "hair" \
  --edit-prompt "make the hair red"

# 步骤2：小帧数测试
python scripts/edit_video_distributed.py \
  --input examples/rgb.mp4 \
  --output outputs/test_30frames.mp4 \
  --mask-prompt "hair" \
  --edit-prompt "make the hair red" \
  --max-frames 30

# 步骤3：完整视频
python scripts/edit_video_distributed.py \
  --input examples/rgb.mp4 \
  --output outputs/final.mp4 \
  --mask-prompt "hair" \
  --edit-prompt "make the hair red"
```

---

## 总结

**最简单的调试方式：**

```bash
python scripts/edit_video_distributed.py \
  --input your_video.mp4 \
  --output outputs/result.mp4 \
  --mask-prompt "要编辑的目标" \
  --edit-prompt "编辑指令" \
  --max-frames 30
```

就这么简单！不需要UI，直接命令行运行即可。

