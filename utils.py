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
        print("[ensure_hf_login] Checking Hugging Face login status...")
        whoami()
        print("[ensure_hf_login] Already logged into Hugging Face")
    except Exception as e:
        print("[ensure_hf_login] Not logged in. Exception:", e)
        print("[ensure_hf_login] Logging into Hugging Face...")
        token = os.environ.get("HF_TOKEN")
        print(f"[ensure_hf_login] HF_TOKEN from env: {'FOUND' if token else 'NOT FOUND'}")
        if not token:
            try:
                print("[ensure_hf_login] Trying to read token from hf_token.txt...")
                with open("hf_token.txt", "r") as f:
                    token = f.read().strip()
                print("[ensure_hf_login] Token read from hf_token.txt")
            except Exception as e:
                print("[ensure_hf_login] Could not read HF token from hf_token.txt:", e)
                token = None
        else:
            print("[ensure_hf_login] Using token from environment variable")
        login(token=token)
        print("[ensure_hf_login] Login attempted with token:", "PRESENT" if token else "MISSING")

def cleanup_temp_files(folder: str) -> None:
    """Clean up temporary files and folders."""
    if os.path.exists(folder):
        shutil.rmtree(folder)

def optimise_image_condition(image: Image.Image, delta=[0,0,0]) -> Image.Image:
    """
    Remove the white space from the image by cropping to the bounding box of non-white pixels.

    Args:
        image (PIL.Image.Image): The input image to be cropped.
        delta (list, optional): A list of three integers, used to store cropping offsets. 
            The function updates delta[1] and delta[2] with the y and x offsets (in multiples of 16) 
            used for cropping. Default is [0, 0, 0].

    Returns:
        PIL.Image.Image: The cropped image with white space removed.
        list: The updated delta list with cropping offsets.
    """
    # Use thresholding to detect white background and find bounding box for non-white part of image
    width, height = image.size
    arr = np.array(image)
    if arr.shape[-1] == 4:
        rgb = arr[..., :3]
    else:
        rgb = arr

    # Define a threshold for "white" (tolerate slight off-white)
    threshold = 240
    # Create mask: True where pixel is NOT white
    nonwhite_mask = np.any(rgb < threshold, axis=-1)

    # Find bounding box of non-white region
    coords = np.argwhere(nonwhite_mask)
    if coords.size == 0:
        # No non-white pixels, return original image
        return image, delta

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1  # +1 because slicing is exclusive

    # Add padding to the image
    x0 = max(0, x0 - 16)
    y0 = max(0, y0 - 16)
    y1 = min(height, y1 + 16)
    x1 = min(width, x1 + 16)

    
    # Apply delta if provided
    x0 = x0 //16 * 16
    y0 = y0 //16 * 16



    x1 = x0 + (x1-x0)//16 * 16
    y1 = y0 + (y1-y0)//16 * 16


    delta[1] = y0//16
    delta[2] = x0//16
    # Crop and return
    return image.crop((x0, y0, x1, y1)), delta
