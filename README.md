# UR-Edit: Unstructured Regional Image & Video Editing

**UR-Edit** 是一个致力于解决高精度、非结构化区域性图像与视频编辑问题的开源项目。通过结合 **SAM3 (Segment Anything Model 3)** 的强泛化分割能力与 **Instruct-Pix2Pix** 的指令式生成能力，实现了对视频中任意非规则区域（Unstructured Regions）的精准锁定与风格化编辑。

核心特性：
- **非结构化区域感知**：利用 SAM3 对任意形状、任意类别的非结构化目标进行像素级分割与全视频跟踪。
- **区域性精准编辑**：基于 Mask 的 Crop-Edit-Paste 策略，仅对感兴趣区域进行高分辨率重绘，完美保留背景上下文。
- **指令驱动**：支持自然语言指令（如 "make it red", "turn into cyborg"），无需繁琐的掩码绘制。
- **分布式高清处理**：支持 8卡并行加速，处理 4K/8K 高清视频流。

---

## 1. 环境配置 (Installation)

本项目推荐使用 **Python 3.12** 和 **CUDA 12.x** 环境。

### 1.1 创建 Conda 环境
```bash
conda create -n sam3 python=3.12
conda activate sam3
```

### 1.2 安装 PyTorch
请根据您的 CUDA 版本安装 PyTorch（推荐 2.7+）：

### 1.3 安装 SAM3
进入 `sam3` 目录并安装：
```bash
cd sam3
pip install -e .
cd ..
```

### 1.4 安装 Instruct-Pix2Pix 及其他依赖
安装 `diffusers`, `transformers`, `accelerate` 以及图像处理库：
```bash
pip install diffusers["torch"] transformers accelerate
pip install opencv-python imageio imageio-ffmpeg matplotlib
```

### 1.5 版本特别说明 (NumPy)
```bash
pip install "numpy==1.26"
```

---

## 2. 模型准备 (Model Preparation)

请确保以下模型文件已下载并放置在正确位置。

### 2.1 SAM3 模型
- **路径**: `./pretrained_models/sam3/sam3.pt`
- **来源**: 请从 SAM3 官方仓库或 HuggingFace 下载。

### 2.2 Instruct-Pix2Pix 模型 (Diffusers 格式)
本项目使用 Diffusers 格式加载模型以支持离线和分布式运行。
- **路径**: `./instruct-pix2pix/diffusers_model`
- **获取方式**:
  如果您只有 `.ckpt` 文件，请使用根目录下的转换脚本：
  ```bash
  python scripts/convert_ckpt.py \
    --checkpoint_path /path/to/instruct-pix2pix-00-22000.ckpt \
    --dump_path ./instruct-pix2pix/diffusers_model
  ```
  *(注意：转换时需要加载 CLIP 模型，如果无法联网，请确保本地有 `clip-vit-large-patch14` 权重并修改转换脚本指向本地)*

---

## 3. 单帧/图片编辑测试 (Single Frame Editing)

在运行全视频编辑前，强烈建议先在单张图片或视频第一帧上调试 Prompt 和参数。

**脚本**: `scripts/single_frame_edit.py`

### 3.1 运行命令示例
```bash
python scripts/single_frame_edit.py \
  --input "examples/b64b82ca-afb4-4894-856f-6dd1a7618492.png" \
  --mask-prompt "yellow dress" \
  --edit-prompt "make the yellow dress red, keep its origin style" \
  --image-guidance-scale 1.5 \
  --guidance-scale 9.0 \
  --erode-kernel 15
```

### 3.2 关键参数说明
| 参数 | 说明 | 推荐值 |
| :--- | :--- | :--- |
| `--input` | 输入图片路径或视频路径 | 支持 .jpg, .png, .mp4 |
| `--mask-prompt` | **SAM3 分割提示词**，描述要编辑的物体 | 如 "hair", "dress", "car" |
| `--edit-prompt` | **编辑指令**，描述要进行的操作 | 如 "make it red", "turn into cyborg" |
| `--image-guidance-scale` | **原图保留程度**。值越低变化越大，值越高越像原图 | **1.2 - 1.5** (改色/微调)<br>**0.8 - 1.1** (结构改变/换Logo) |
| `--guidance-scale` | **文本指令依从度**。值越高越听指令 | **7.5 - 12.0** |
| `--erode-kernel` | **Mask 腐蚀大小**。用于收缩 Mask 边缘，防止误改背景/人脸 | **5 - 15** |

---

## 4. 全视频分布式编辑 (Distributed Video Editing)

确认单帧效果后，使用此脚本进行全视频处理。脚本会自动利用所有可用 GPU (如 8x3090) 进行加速。

**脚本**: `scripts/edit_video_distributed.py`

### 4.1 运行命令示例
```bash
python scripts/edit_video_distributed.py \
  --input examples/rgb.mp4 \
  --output outputs/final_video.mp4 \
  --mask-prompt "hair" \
  --edit-prompt "make the hair curly and black" \
  --image-guidance-scale 1.5 \
  --guidance-scale 9.0 \
  --erode-kernel 10 \
  --max-frames 60
```

### 4.2 特性说明
*   **SAM3 跟踪策略**: 采用 "Text Init + Points Fix" 双重策略，确保第一帧 Mask 极其精准，并传播到全视频。
*   **Crop-Edit-Paste**: 仅裁剪目标区域进行高分辨率编辑，再通过模糊遮罩贴回，避免背景闪烁和画质损失。
*   **离线运行**: 设置了 `local_files_only=True`，无需联网连接 HuggingFace。

---

## 5. 常见问题 (FAQ)

1.  **ImportError: cannot import name '...' from 'diffusers'**
    *   检查 `diffusers` 版本，建议更新到最新版。
    *   检查 Python 环境是否激活正确。

2.  **Mask 覆盖到了不该覆盖的地方（如人脸）**
    *   **调大 `--erode-kernel`**：增加腐蚀力度，让 Mask 向内收缩。
    *   **优化 `--mask-prompt`**：使用更精确的词，或在 SAM3 提示中添加负面提示（目前脚本默认只支持正面提示，可通过代码扩展）。

3.  **编辑后的物体像“补丁”一样，背景不融合**
    *   这是 Crop 模式的常见问题。
    *   **解决方案**：确保脚本启用了 **Mask Fusion** (即 Paste with Mask)。目前的 `single_frame_edit.py` 和 `edit_video_distributed.py` 均已包含此逻辑。

4.  **OOM (显存不足)**
    *   虽然已开启 `device_map="balanced"`，但 SAM3 可能会占用 GPU 0。
    *   尝试减小 `Crop` 时的分辨率限制（目前代码中 Resize 到 512）。
