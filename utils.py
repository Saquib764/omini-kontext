from rembg import remove
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageFilter
import io
import os
import subprocess
import time
import requests
from io import BytesIO
import base64
from pathlib import Path
from huggingface_hub import login, whoami
from typing import Union, Optional, Tuple, List
import shutil


def remove_background(
    image: Image.Image,
    cropped: bool = True,
    padding: int = 10
) -> Image.Image:
    """Remove background and optionally crop to content."""
    # Convert the PIL image to bytes
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    
    # Remove background
    output_bytes = remove(buffer.getvalue())
    output_image = Image.open(io.BytesIO(output_bytes)).convert("RGBA")
    
    if cropped:
        # Crop to content with padding
        alpha = output_image.split()[3]
        bbox = alpha.getbbox()
        if bbox:
            left, upper, right, lower = bbox
            width, height = output_image.size

            # Apply padding within bounds
            left = max(0, left - padding)
            upper = max(0, upper - padding)
            right = min(width, right + padding)
            lower = min(height, lower + padding)

            output_image = output_image.crop((left, upper, right, lower))
    
    return output_image


def download_weights(url: str, dest: Path) -> None:
    """Download model weights from URL to destination."""
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

def ensure_hf_login() -> None:
    """Ensure logged into HuggingFace."""
    try:
        whoami()
        print("Already logged into Hugging Face")
    except Exception:
        print("Logging into Hugging Face...")
        token = os.environ.get("HF_TOKEN")
        if not token:
            try:
                with open("hf_token.txt", "r") as f:
                    token = f.read().strip()
            except Exception as e:
                print("Could not read HF token from hf_token.txt:", e)
                token = None
        login(token=token)

def cleanup_temp_files(folder: str) -> None:
    """Clean up temporary files and folders."""
    if os.path.exists(folder):
        shutil.rmtree(folder)
