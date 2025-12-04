import torch
import os
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_from_original_stable_diffusion_ckpt
from transformers import CLIPTokenizer, CLIPTextModel

checkpoint_path = "/home/zmh/SAM3-Video-Editor/instruct-pix2pix/checkpoints/instruct-pix2pix-00-22000.ckpt"
original_config_file = "/home/zmh/SAM3-Video-Editor/instruct-pix2pix/configs/generate.yaml"
dump_path = "/home/zmh/SAM3-Video-Editor/instruct-pix2pix/diffusers_model"
clip_path = "/home/zmh/SAM3-Video-Editor/pretrained_models/clip-vit-large-patch14"

print(f"Loading local CLIP from {clip_path}...")
try:
    tokenizer = CLIPTokenizer.from_pretrained(clip_path, local_files_only=True)
    text_encoder = CLIPTextModel.from_pretrained(clip_path, local_files_only=True)
    print("Local CLIP loaded successfully!")
except Exception as e:
    print(f"Error loading CLIP: {e}")
    exit(1)

print(f"Converting {checkpoint_path}...")
print(f"Using config {original_config_file}...")

# 尝试转换
pipe = download_from_original_stable_diffusion_ckpt(
    checkpoint_path_or_dict=checkpoint_path,
    original_config_file=original_config_file,
    image_size=512,
    prediction_type="epsilon",
    extract_ema=True,
    scheduler_type="euler-ancestral",
    num_in_channels=8,
    upcast_attention=False,
    device="cpu",
    from_safetensors=False,
    load_safety_checker=False, # 关键修改：禁用安全检查器下载
    tokenizer=tokenizer,      # 传入本地加载的 tokenizer
    text_encoder=text_encoder # 传入本地加载的 text_encoder
)

print(f"Saving to {dump_path}...")
pipe.save_pretrained(dump_path)
print("Conversion successful!")
