import os
import io
import base64
import hashlib
import json
from typing import Tuple
import random
import time
import math

import torch
import numpy as np
from PIL import Image

from server import PromptServer

# Global storage for reference settings
editor_scales = {}

# Constants
MAX_CANVAS_SIZE = 500
DEFAULT_IMAGE_SIZE = 512

def _tensor_to_pil(img: torch.Tensor) -> Image.Image:
    """Convert IMAGE tensor [B, H, W, C] with values [0,1] to PIL Image."""
    if img.dim() == 3:
        img = img.unsqueeze(-1)
    if img.dim() != 4:
        raise ValueError("IMAGE tensor must be [B,H,W,C]")
    
    img0 = img[0].detach().cpu().clamp(0, 1).numpy()
    h, w, c = img0.shape
    
    if c == 3:
        mode = "RGB"
    elif c == 1:
        mode = "L"
        img0 = img0.squeeze(-1)
    else:
        mode = "RGBA"
    
    arr = (img0 * 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode=mode)

def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert PIL (RGB/RGBA) to IMAGE tensor [1,H,W,C], float32 in [0,1]."""
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA")
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...]

def _png_data_url(pil_img: Image.Image) -> str:
    """Convert PIL image to PNG data URL."""
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

class OminiKontextEditor:
    """
    A super-simple image editor node:
      - UI lets user position and scale reference images
      - Reference settings are saved and can trigger re-execution
      - On execute, we display the base and reference images
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "background": ("IMAGE",),
                "subject": ("IMAGE",),
                "subject_mask": ("MASK",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    FUNCTION = "do_composite"
    CATEGORY = "image"

    def _push_bg(self, unique_id: str, background: Image.Image, subject: Image.Image):
        """Send images to browser widget."""
        try:
            data_url = _png_data_url(background.convert("RGBA"))
            subject_data_url = _png_data_url(subject.convert("RGBA"))
            
            # Get saved subject settings if they exist
            subject_settings = editor_scales.get(str(unique_id))
            
            PromptServer.instance.send_sync(
                "simpledraw_bg",
                {
                    "unique_id": str(unique_id), 
                    "background": data_url, 
                    "subject": subject_data_url,
                    "subject_settings": subject_settings
                },
            )
        except Exception as e:
            pass

    def _composite_reference_image(self, background_img: Image.Image, subject_img: Image.Image, subject_settings: dict) -> Tuple[Image.Image, Image.Image]:
        """Composite subject image onto background image according to settings."""
        # Create output images
        white_bg = Image.new('RGBA', background_img.size, (255, 255, 255, 255))
        base_composite = background_img.copy()
        
        if not subject_settings:
            return white_bg.convert("RGB"), base_composite.convert("RGB")
        
        # Calculate the actual position and scale in the final image
        canvas_scale = subject_settings.get('overallScale', 1.0)
        
        # Scale the subject image according to saved scale
        ref_width = int(subject_img.width * subject_settings.get('scaleX', 1.0) / canvas_scale)
        ref_height = int(subject_img.height * subject_settings.get('scaleY', 1.0) / canvas_scale)
        
        # Resize subject image
        scaled_ref = subject_img.resize((ref_width, ref_height), Image.LANCZOS)
        
        # Apply rotation if specified
        angle = subject_settings.get('angle', 0)
        if angle != 0:
            # Expand the image to accommodate rotation
            expanded_size = int(max(ref_width, ref_height) * 1.5)
            expanded_img = Image.new('RGBA', (expanded_size, expanded_size), (0, 0, 0, 0))
            
            # Center the scaled reference image on the expanded canvas
            paste_x = (expanded_size - ref_width) // 2
            paste_y = (expanded_size - ref_height) // 2
            expanded_img.paste(scaled_ref, (paste_x, paste_y))
            
            # Rotate the expanded image
            rotated_img = expanded_img.rotate(-angle, expand=True, resample=Image.BICUBIC)
            scaled_ref = rotated_img
            
        
        # Ensure the scaled reference has proper alpha channel
        if scaled_ref.mode != 'RGBA':
            scaled_ref = scaled_ref.convert('RGBA')
        
        # Calculate position in the final image coordinates
        left = int(subject_settings.get('left', 0) / canvas_scale)
        top = int(subject_settings.get('top', 0) / canvas_scale)
        
        # Adjust position for rotated image to maintain center alignment
        if angle != 0:
            # Get the rotated image dimensions
            angle_rad = angle * math.pi / 180
            rotated_width, rotated_height = scaled_ref.size
            # Adjust position to keep the center of the rotated image at the same point

            diag = math.sqrt(ref_width**2 + ref_height**2)
            corner_angle = math.atan2(ref_width, ref_height)

            rotation_x = diag * math.sin(corner_angle - angle_rad)
            rotation_y = diag * math.cos(corner_angle - angle_rad)
            left = left - rotated_width // 2 + int(rotation_x // 2)
            top = top - rotated_height // 2 + int(rotation_y // 2)
        
        # Create a proper mask for compositing
        # If the image has transparency, use it as the mask
        if scaled_ref.mode == 'RGBA':
            # Extract alpha channel as mask
            alpha_mask = scaled_ref.split()[-1]  # Get alpha channel
            # Ensure mask is in 'L' mode (grayscale)
            if alpha_mask.mode != 'L':
                alpha_mask = alpha_mask.convert('L')
        else:
            # If no alpha channel, create a solid mask
            alpha_mask = Image.new('L', scaled_ref.size, 255)
        
        # Paste the scaled subject image at the calculated position
        try:
            white_bg.paste(scaled_ref, (left, top), alpha_mask)
            base_composite.paste(scaled_ref, (left, top), alpha_mask)
        except Exception as e:
            # Fallback: paste without mask
            white_bg.paste(scaled_ref, (left, top))
            base_composite.paste(scaled_ref, (left, top))
        
        return white_bg.convert("RGB"), base_composite.convert("RGB")

    def do_composite(self, background: torch.Tensor, subject: torch.Tensor, subject_mask: torch.Tensor, unique_id):
        """Main composite function."""
        # Convert input tensor to PIL
        base = _tensor_to_pil(background)
        subject_img = _tensor_to_pil(subject)
        subject_mask_img = _tensor_to_pil(1 - subject_mask)
        
        # Combine subject with mask if sizes match
        if subject_img.size == subject_mask_img.size:
            try:
                # Ensure subject is in RGB mode for splitting
                if subject_img.mode != 'RGB':
                    subject_img = subject_img.convert('RGB')
                
                # Ensure mask is in 'L' mode (grayscale)
                if subject_mask_img.mode != 'L':
                    subject_mask_img = subject_mask_img.convert('L')
                
                # Split RGB channels and combine with mask
                r, g, b = subject_img.split()
                subject_img = Image.merge("RGBA", (r, g, b, subject_mask_img))
            except Exception as e:
                # Fallback: convert subject to RGBA without mask
                subject_img = subject_img.convert('RGBA')
        else:
            # If sizes don't match, ensure subject is in RGBA mode
            if subject_img.mode != 'RGBA':
                subject_img = subject_img.convert('RGBA')

        # Send images to browser widget
        self._push_bg(unique_id, base, subject_img)
        
        # Get saved subject settings
        subject_settings = None
        while subject_settings is None:
            subject_settings = editor_scales.get(str(unique_id))
            time.sleep(0.4)

        # Composite images
        white_composite, base_composite = self._composite_reference_image(base, subject_img, subject_settings)
        
        return (_pil_to_tensor(white_composite), _pil_to_tensor(base_composite))

    @classmethod
    def IS_CHANGED(cls, background, subject, subject_mask, unique_id):
        """Re-run when input image content changes or reference settings change."""
        h = hashlib.sha256()

        # Hash input tensors
        for tensor, name in [(background, "base"), (subject, "subject"), (subject_mask, "mask")]:
            try:
                arr = tensor[0].detach().cpu().clamp(0, 1).numpy()
                h.update(arr.tobytes())
            except Exception:
                h.update(f"no_{name}_image".encode())

        # Hash reference settings if they exist
        if str(unique_id) in editor_scales:
            ref_settings = editor_scales[str(unique_id)]
            settings_str = json.dumps(ref_settings, sort_keys=True)
            h.update(settings_str.encode("utf-8"))
        else:
            h.update(str(random.random()).encode("utf-8"))

        return h.hexdigest()

    @classmethod
    def VALIDATE_INPUTS(cls, background, subject, subject_mask, unique_id):
        return True


def register_upload_api():
    """Register custom upload API endpoint for reference settings"""
    try:
        from server import PromptServer
        from aiohttp import web
        import json
        
        server = PromptServer.instance
        
        @server.routes.post("/omini_kontext_editor/update_subject_settings")
        async def handle_subject_settings_update(request):
            """Handle subject settings updates from the editor"""
            try:
                data = await request.json()
                unique_id = data.get('unique_id')
                settings = data.get('settings')
                
                if unique_id is None:
                    return web.json_response({"error": "No unique_id provided"}, status=400)
                
                # Handle settings removal
                if settings is None:
                    if str(unique_id) in editor_scales:
                        del editor_scales[str(unique_id)]
                    
                    return web.json_response({
                        "status": "success",
                        "message": "Subject settings cleared successfully",
                        "unique_id": unique_id
                    })
                
                # Store subject settings
                editor_scales[str(unique_id)] = {
                    'left': settings.get('left'),
                    'top': settings.get('top'),
                    'scaleX': settings.get('scaleX'),
                    'scaleY': settings.get('scaleY'),
                    'angle': settings.get('angle'),
                    'canvasWidth': settings.get('canvasWidth'),
                    'canvasHeight': settings.get('canvasHeight'),
                    'overallScale': settings.get('overallScale')
                }
                
                return web.json_response({
                    "status": "success",
                    "message": "Subject settings updated successfully",
                    "unique_id": unique_id
                })
                
            except Exception as e:
                error_msg = f"Error updating reference settings: {e}"
                return web.json_response({"error": error_msg}, status=500)
        
    except Exception as e:
        pass


# ComfyUI registry
NODE_CLASS_MAPPINGS = {
    "OminiKontextEditor": OminiKontextEditor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OminiKontextEditor": "Omini Kontext Editor",
}

# Register the upload API when the module is loaded
try:
    register_upload_api()
except Exception as e:
    pass
