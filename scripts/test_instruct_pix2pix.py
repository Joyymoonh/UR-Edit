#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 InstructPix2Pix 图像编辑
"""

import sys
import os
import torch
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from sam3_masking import SAM3Predictor


def test_instruct_pix2pix():
    """测试 InstructPix2Pix 图像编辑"""
    
    print("=" * 70)
    print("测试 InstructPix2Pix 图像编辑")
    print("=" * 70)
    
    # 检查设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n设备: {device}")
    
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 1. 加载 InstructPix2Pix
    print("\n[1/3] 加载 InstructPix2Pix 模型...")
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None
    ).to(device)
    
    if device == "cuda":
        pipe.enable_attention_slicing()
    
    print("✓ 模型加载成功")
    
    # 2. 创建测试图像
    print("\n[2/3] 创建测试图像...")
    
    # 如果有测试图像就用，否则创建一个
    test_image_path = "test_image.jpg"
    
    if os.path.exists(test_image_path):
        image = Image.open(test_image_path).convert("RGB")
        print(f"✓ 加载测试图像: {test_image_path}")
    else:
        # 创建一个简单的测试图像
        import numpy as np
        img_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        image = Image.fromarray(img_array)
        image.save(test_image_path)
        print(f"✓ 创建测试图像: {test_image_path}")
    
    # 调整大小
    image = image.resize((512, 512))
    
    # 3. 测试编辑
    print("\n[3/3] 测试图像编辑...")
    
    test_prompts = [
        "make the hair red",
        "add sunglasses",
        "make it smile"
    ]
    
    os.makedirs("outputs/test", exist_ok=True)
    
    # 保存原图
    image.save("outputs/test/0_original.jpg")
    print("  保存原图: outputs/test/0_original.jpg")
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n  测试 {i}/{len(test_prompts)}: '{prompt}'")
        
        try:
            edited = pipe(
                prompt=prompt,
                image=image,
                num_inference_steps=20,  # 快速测试
                image_guidance_scale=1.5,
                guidance_scale=7.5
            ).images[0]
            
            output_path = f"outputs/test/{i}_{prompt.replace(' ', '_')}.jpg"
            edited.save(output_path)
            print(f"  ✓ 保存结果: {output_path}")
            
        except Exception as e:
            print(f"  ✗ 编辑失败: {e}")
    
    print("\n" + "=" * 70)
    print("测试完成！")
    print("结果保存在: outputs/test/")
    print("=" * 70)


def test_sam3():
    """测试 SAM3 分割"""
    
    print("\n" + "=" * 70)
    print("测试 SAM3 分割")
    print("=" * 70)
    
    try:
        print("\n[1/2] 初始化 SAM3...")
        sam = SAM3Predictor()
        print("✓ SAM3 初始化成功")
        
        print("\n[2/2] 测试分割...")
        
        # 创建测试图像
        test_image_path = "test_image.jpg"
        if os.path.exists(test_image_path):
            image = Image.open(test_image_path).convert("RGB")
            
            # 测试分割
            mask = sam.generate_mask(image, "person")
            print(f"✓ 分割成功，mask shape: {mask.shape}")
            
            # 保存 mask
            import torchvision
            torchvision.utils.save_image(mask, "outputs/test/sam3_mask.png")
            print("✓ Mask 保存: outputs/test/sam3_mask.png")
        else:
            print("✗ 测试图像不存在，跳过")
        
        print("\n" + "=" * 70)
        print("SAM3 测试完成！")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ SAM3 测试失败: {e}")
        print("请检查 SAM3 模型是否正确安装")


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 20 + "环境测试套件" + " " * 34 + "║")
    print("╚" + "=" * 68 + "╝")
    
    # 测试 1: InstructPix2Pix
    try:
        test_instruct_pix2pix()
    except Exception as e:
        print(f"\n✗ InstructPix2Pix 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试 2: SAM3
    try:
        test_sam3()
    except Exception as e:
        print(f"\n✗ SAM3 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("所有测试完成！")
    print("=" * 70)
