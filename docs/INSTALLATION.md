# 完整安装指南

## 📋 目录
1. [系统要求](#系统要求)
2. [快速安装](#快速安装)
3. [手动安装](#手动安装)
4. [验证安装](#验证安装)
5. [常见问题](#常见问题)

---

## 🖥️ 系统要求

### 硬件要求
- **GPU**: NVIDIA GPU (推荐 RTX 3090 或更高)
  - 最低显存: 12GB (图像编辑)
  - 推荐显存: 24GB (视频编辑)
- **内存**: 32GB RAM (推荐)
- **存储**: 50GB 可用空间

### 软件要求
- **操作系统**: Linux (Ubuntu 20.04+) / Windows 10+ / macOS
- **CUDA**: 11.8 或 12.1 (如果使用 GPU)
- **Conda**: Anaconda 或 Miniconda

---

## ⚡ 快速安装

### 方法 1: 自动安装脚本（推荐）

```bash
cd /home/zmh/SAM3-Video-Editor
chmod +x setup_complete.sh
bash setup_complete.sh
```

脚本会自动完成：
- ✅ 创建 conda 环境
- ✅ 安装 PyTorch (CUDA 11.8)
- ✅ 安装所有依赖
- ✅ 验证安装

**预计时间**: 10-15 分钟

---

## 🔧 手动安装

如果自动脚本失败，可以手动安装：

### 步骤 1: 创建 Conda 环境

```bash
# 创建环境
conda create -n sam3video python=3.9 -y

# 激活环境
conda activate sam3video
```

### 步骤 2: 安装 PyTorch

#### GPU 版本 (CUDA 11.8)
```bash
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
```

#### GPU 版本 (CUDA 12.1)
```bash
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
```

#### CPU 版本
```bash
pip install torch==2.1.0 torchvision==0.16.0
```

### 步骤 3: 安装核心依赖

```bash
# 基础库
pip install numpy==1.24.3
pip install pillow==10.0.0
pip install tqdm==4.66.1
pip install matplotlib==3.7.2

# 视频处理
pip install opencv-python==4.8.1.78
pip install imageio==2.31.5
pip install imageio-ffmpeg==0.4.9
pip install scikit-image==0.21.0
```

### 步骤 4: 安装 Diffusion 模型库

```bash
# Diffusers 和相关
pip install diffusers==0.21.4
pip install transformers==4.35.0
pip install accelerate==0.24.1
pip install safetensors==0.4.0

# CLIP
pip install ftfy regex
pip install git+https://github.com/openai/CLIP.git
```

### 步骤 5: 安装 SAM3 依赖

```bash
pip install timm==0.9.12
pip install huggingface_hub==0.19.4
pip install iopath==0.1.10
```

### 步骤 6: 验证安装

```bash
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
"
```

---

## ✅ 验证安装

### 1. 检查 Python 包

```bash
conda activate sam3video

python << EOF
import torch
import cv2
import diffusers
import transformers
import clip

print("✓ PyTorch:", torch.__version__)
print("✓ CUDA:", torch.cuda.is_available())
print("✓ OpenCV:", cv2.__version__)
print("✓ Diffusers:", diffusers.__version__)
print("✓ Transformers:", transformers.__version__)
print("✓ CLIP: OK")
EOF
```

### 2. 测试 SAM3

```bash
python << EOF
import sys
sys.path.insert(0, 'sam3')
from sam3 import build_sam3

print("✓ SAM3 模块导入成功")
EOF
```

### 3. 测试 InstructPix2Pix

```bash
python test_instruct_pix2pix.py
```

### 4. 测试完整流程

```bash
# 准备一个测试视频（10秒以内）
python main.py \
    --video_path test_video.mp4 \
    --mask_prompt "hair" \
    --edit_prompt "red hair" \
    --output_path test_output.mp4 \
    --max_frames 50
```

---

## 🐛 常见问题

### 问题 1: CUDA 不可用

**症状**:
```
torch.cuda.is_available() = False
```

**解决方案**:
```bash
# 检查 NVIDIA 驱动
nvidia-smi

# 重新安装 PyTorch (确保 CUDA 版本匹配)
pip uninstall torch torchvision
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
```

### 问题 2: 显存不足 (OOM)

**症状**:
```
RuntimeError: CUDA out of memory
```

**解决方案**:
```bash
# 方案 1: 降低分辨率
python main.py --video_path input.mp4 ... --max_frames 50

# 方案 2: 启用内存优化
python main.py --video_path input.mp4 ... --enable_attention_slicing --enable_xformers

# 方案 3: 使用 CPU (慢)
python main.py --video_path input.mp4 ... --device cpu
```

### 问题 3: OpenCV 导入错误

**症状**:
```
ImportError: libGL.so.1: cannot open shared object file
```

**解决方案**:
```bash
# Ubuntu/Debian
sudo apt-get install libgl1-mesa-glx

# 或使用 headless 版本
pip uninstall opencv-python
pip install opencv-python-headless
```

### 问题 4: SAM3 模型未找到

**症状**:
```
FileNotFoundError: pretrained_models/sam3/sam3.pt
```

**解决方案**:
```bash
# 检查模型文件
ls -lh pretrained_models/sam3/sam3.pt

# 如果不存在，从原项目复制
cp /home/zmh/ReWarp-CLIP/pretrained_models/sam3/sam3.pt pretrained_models/sam3/
```

### 问题 5: Diffusers 模型下载慢

**症状**:
模型下载卡住或很慢

**解决方案**:
```bash
# 设置 Hugging Face 镜像
export HF_ENDPOINT=https://hf-mirror.com

# 或手动下载模型
huggingface-cli download runwayml/stable-diffusion-inpainting
```

### 问题 6: xformers 安装失败

**症状**:
```
ERROR: Could not build wheels for xformers
```

**解决方案**:
```bash
# xformers 是可选的，可以不安装
# 如果需要，使用预编译版本
pip install xformers==0.0.22
```

---

## 📦 完整依赖列表

### 核心依赖
```
torch==2.1.0
torchvision==0.16.0
numpy==1.24.3
pillow==10.0.0
```

### 视频处理
```
opencv-python==4.8.1.78
imageio==2.31.5
imageio-ffmpeg==0.4.9
scikit-image==0.21.0
```

### Diffusion 模型
```
diffusers==0.21.4
transformers==4.35.0
accelerate==0.24.1
safetensors==0.4.0
clip (from git)
```

### SAM3
```
timm==0.9.12
huggingface_hub==0.19.4
iopath==0.1.10
```

### 工具
```
tqdm==4.66.1
matplotlib==3.7.2
```

---

## 🚀 性能优化

### 1. 启用 xformers (可选)

```bash
pip install xformers==0.0.22

# 使用时添加参数
python main.py ... --enable_xformers
```

### 2. 使用 Flash Attention (可选)

```bash
pip install flash-attn

# 会自动使用
```

### 3. 使用混合精度

```bash
# 默认使用 float16
python main.py ... --dtype float16
```

---

## 📊 环境对比

| 环境 | PyTorch | CUDA | 显存占用 | 速度 |
|------|---------|------|---------|------|
| 最小配置 | 2.0.0 | 11.7 | 8GB | 慢 |
| 推荐配置 | 2.1.0 | 11.8 | 12GB | 中 |
| 最佳配置 | 2.1.0 | 12.1 | 24GB | 快 |

---

## 🔄 更新环境

```bash
# 激活环境
conda activate sam3video

# 更新所有包
pip install --upgrade -r requirements.txt

# 或更新特定包
pip install --upgrade diffusers transformers
```

---

## 🗑️ 卸载

```bash
# 删除 conda 环境
conda env remove -n sam3video

# 删除项目文件
rm -rf /home/zmh/SAM3-Video-Editor
```

---

## 📞 获取帮助

如果遇到问题：
1. 查看本文档的常见问题部分
2. 检查 GitHub Issues
3. 提交新的 Issue（附上错误信息和环境信息）

获取环境信息：
```bash
conda activate sam3video
python -c "
import sys
import torch
print('Python:', sys.version)
print('PyTorch:', torch.__version__)
print('CUDA:', torch.version.cuda)
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')
"
```
