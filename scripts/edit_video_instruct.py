#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
示例：使用SAM3 + Instruct-Pix2Pix编辑视频
"""

import sys
import os
import argparse

# 将项目根目录添加到 python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.sam3_video_editor import SAM3InstructVideoEditor


def main():
    parser = argparse.ArgumentParser(description="SAM3 + Instruct-Pix2Pix 视频编辑")
    
    # 必需参数
    parser.add_argument("--input", "-i", required=True, help="输入视频路径")
    parser.add_argument("--output", "-o", required=True, help="输出视频路径")
    parser.add_argument("--mask-prompt", "-m", required=True, 
                       help="SAM3分割提示词（如 'person', 'hair', 'face'）")
    parser.add_argument("--edit-prompt", "-e", required=True,
                       help="Instruct-Pix2Pix编辑指令（如 'make the hair red'）")
    
    # 可选参数
    parser.add_argument("--max-frames", type=int, default=None,
                       help="最大处理帧数（测试用）")
    parser.add_argument("--steps", type=int, default=50,
                       help="Instruct-Pix2Pix推理步数（默认50）")
    parser.add_argument("--no-smoothing", action="store_true",
                       help="禁用时序平滑")
    parser.add_argument("--smoothing-window", type=int, default=5,
                       help="平滑窗口大小（默认5）")
    parser.add_argument("--smoothing-alpha", type=float, default=0.3,
                       help="平滑强度 0-1（默认0.3）")
    parser.add_argument("--device", default="cuda",
                       help="设备（cuda或cpu）")
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"错误: 输入视频不存在: {args.input}")
        return
    
    # 初始化编辑器
    print("\n初始化编辑器...")
    editor = SAM3InstructVideoEditor(device=args.device)
    
    # 编辑视频
    editor.edit_video(
        video_path=args.input,
        mask_prompt=args.mask_prompt,
        edit_prompt=args.edit_prompt,
        output_path=args.output,
        max_frames=args.max_frames,
        num_inference_steps=args.steps,
        apply_smoothing=not args.no_smoothing,
        smoothing_window=args.smoothing_window,
        smoothing_alpha=args.smoothing_alpha
    )


def example_usage():
    """打印使用示例"""
    print("\n" + "=" * 70)
    print("使用示例")
    print("=" * 70)
    
    examples = [
        {
            "name": "发型编辑 - 变红发",
            "cmd": """python examples/edit_video_instruct.py \\
    --input input.mp4 \\
    --output output_red_hair.mp4 \\
    --mask-prompt "hair" \\
    --edit-prompt "make the hair red" \\
    --max-frames 100 \\
    --steps 50"""
        },
        {
            "name": "发型编辑 - 变爆炸头",
            "cmd": """python examples/edit_video_instruct.py \\
    --input input.mp4 \\
    --output output_afro.mp4 \\
    --mask-prompt "hair" \\
    --edit-prompt "give the person a big curly black afro hairstyle" \\
    --steps 50"""
        },
        {
            "name": "人物编辑 - 添加墨镜",
            "cmd": """python examples/edit_video_instruct.py \\
    --input input.mp4 \\
    --output output_sunglasses.mp4 \\
    --mask-prompt "face" \\
    --edit-prompt "add sunglasses" \\
    --steps 50"""
        },
        {
            "name": "服装编辑 - 变成西装",
            "cmd": """python examples/edit_video_instruct.py \\
    --input input.mp4 \\
    --output output_suit.mp4 \\
    --mask-prompt "person" \\
    --edit-prompt "make the person wear a formal black suit" \\
    --steps 50"""
        },
        {
            "name": "快速测试（只处理前30帧）",
            "cmd": """python examples/edit_video_instruct.py \\
    --input input.mp4 \\
    --output test_output.mp4 \\
    --mask-prompt "hair" \\
    --edit-prompt "make the hair blue" \\
    --max-frames 30 \\
    --steps 20"""
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n示例 {i}: {example['name']}")
        print("-" * 70)
        print(example['cmd'])
    
    print("\n" + "=" * 70)
    print("参数说明")
    print("=" * 70)
    print("""
--input, -i          输入视频路径
--output, -o         输出视频路径
--mask-prompt, -m    SAM3分割提示词（描述要编辑的区域）
--edit-prompt, -e    编辑指令（描述如何编辑）
--max-frames         最大处理帧数（用于快速测试）
--steps              推理步数（越大质量越好但越慢，推荐20-100）
--no-smoothing       禁用时序平滑（可能导致闪烁）
--smoothing-window   平滑窗口大小（默认5帧）
--smoothing-alpha    平滑强度（0-1，默认0.3）
--device             设备（cuda或cpu）
    """)
    
    print("\n" + "=" * 70)
    print("提示词建议")
    print("=" * 70)
    print("""
Mask Prompt（分割提示词）:
- "person"     - 整个人
- "hair"       - 头发
- "face"       - 脸部
- "upper body" - 上半身
- "clothes"    - 衣服

Edit Prompt（编辑指令）:
- "make the hair [color]"                    - 改变发色
- "give the person a [hairstyle] hairstyle"  - 改变发型
- "add [accessory]"                          - 添加配饰
- "make the person wear [clothing]"          - 改变服装
- "make it [adjective]"                      - 添加风格

注意：
1. 编辑指令要清晰具体
2. 使用英文描述
3. 避免过于复杂的指令
4. 可以多次尝试不同的参数
    """)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # 没有参数，显示使用示例
        example_usage()
    else:
        # 有参数，执行编辑
        main()
