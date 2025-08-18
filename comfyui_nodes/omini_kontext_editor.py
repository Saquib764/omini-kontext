import os
import io
import base64
import hashlib
import json
from typing import Tuple
import random
import time

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
                "base_image": ("IMAGE",),
                "reference_image": ("IMAGE",),
                "reference_mask": ("MASK",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    FUNCTION = "do_composite"
    CATEGORY = "image"

    def _push_bg(self, unique_id: str, base_image: Image.Image, reference_image: Image.Image):
        """Send images to browser widget."""
        try:
            data_url = _png_data_url(base_image.convert("RGBA"))
            reference_data_url = _png_data_url(reference_image.convert("RGBA"))
            
            # Get saved reference settings if they exist
            reference_settings = editor_scales.get(str(unique_id))
            
            PromptServer.instance.send_sync(
                "simpledraw_bg",
                {
                    "unique_id": str(unique_id), 
                    "base_image": data_url, 
                    "reference_image": reference_data_url,
                    "reference_settings": reference_settings
                },
            )
        except Exception as e:
            pass

    def _composite_reference_image(self, base_img: Image.Image, reference_img: Image.Image, reference_settings: dict) -> Tuple[Image.Image, Image.Image]:
        """Composite reference image onto base image according to settings."""
        # Create output images
        white_bg = Image.new('RGBA', base_img.size, (255, 255, 255, 255))
        base_composite = base_img.copy()
        
        if not reference_settings:
            return white_bg.convert("RGB"), base_composite.convert("RGB")
        
        # Calculate the actual position and scale in the final image
        canvas_scale = reference_settings.get('overallScale', 1.0)
        
        # Scale the reference image according to saved scale
        ref_width = int(reference_img.width * reference_settings.get('scaleX', 1.0) / canvas_scale)
        ref_height = int(reference_img.height * reference_settings.get('scaleY', 1.0) / canvas_scale)
        
        # Resize reference image
        scaled_ref = reference_img.resize((ref_width, ref_height), Image.LANCZOS)
        
        # Ensure the scaled reference has proper alpha channel
        if scaled_ref.mode != 'RGBA':
            scaled_ref = scaled_ref.convert('RGBA')
        
        # Calculate position in the final image coordinates
        left = int(reference_settings.get('left', 0) / canvas_scale)
        top = int(reference_settings.get('top', 0) / canvas_scale)
        
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
        
        # Paste the scaled reference image at the calculated position
        try:
            white_bg.paste(scaled_ref, (left, top), alpha_mask)
            base_composite.paste(scaled_ref, (left, top), alpha_mask)
        except Exception as e:
            # Fallback: paste without mask
            white_bg.paste(scaled_ref, (left, top))
            base_composite.paste(scaled_ref, (left, top))
        
        return white_bg.convert("RGB"), base_composite.convert("RGB")

    def do_composite(self, base_image: torch.Tensor, reference_image: torch.Tensor, reference_mask: torch.Tensor, unique_id):
        """Main composite function."""
        # Convert input tensor to PIL
        base = _tensor_to_pil(base_image)
        reference = _tensor_to_pil(reference_image)
        reference_mask = _tensor_to_pil(1 - reference_mask)
        
        # Combine reference with mask if sizes match
        if reference.size == reference_mask.size:
            try:
                # Ensure reference is in RGB mode for splitting
                if reference.mode != 'RGB':
                    reference = reference.convert('RGB')
                
                # Ensure mask is in 'L' mode (grayscale)
                if reference_mask.mode != 'L':
                    reference_mask = reference_mask.convert('L')
                
                # Split RGB channels and combine with mask
                r, g, b = reference.split()
                reference = Image.merge("RGBA", (r, g, b, reference_mask))
            except Exception as e:
                # Fallback: convert reference to RGBA without mask
                reference = reference.convert('RGBA')
        else:
            # If sizes don't match, ensure reference is in RGBA mode
            if reference.mode != 'RGBA':
                reference = reference.convert('RGBA')

        # Send images to browser widget
        self._push_bg(unique_id, base, reference)
        
        # Get saved reference settings
        reference_settings = None
        while reference_settings is None:
            reference_settings = editor_scales.get(str(unique_id))
            time.sleep(0.4)

        # Composite images
        white_composite, base_composite = self._composite_reference_image(base, reference, reference_settings)
        
        return (_pil_to_tensor(white_composite), _pil_to_tensor(base_composite))

    @classmethod
    def IS_CHANGED(cls, base_image, reference_image, reference_mask, unique_id):
        """Re-run when input image content changes or reference settings change."""
        h = hashlib.sha256()

        # Hash input tensors
        for tensor, name in [(base_image, "base"), (reference_image, "reference"), (reference_mask, "mask")]:
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
    def VALIDATE_INPUTS(cls, base_image, reference_image, reference_mask, unique_id):
        return True


def register_upload_api():
    """Register custom upload API endpoint for reference settings"""
    try:
        from server import PromptServer
        from aiohttp import web
        import json
        
        server = PromptServer.instance
        
        @server.routes.post("/omini_kontext_editor/update_reference_settings")
        async def handle_reference_settings_update(request):
            """Handle reference settings updates from the editor"""
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
                        "message": "Reference settings cleared successfully",
                        "unique_id": unique_id
                    })
                
                # Store reference settings
                editor_scales[str(unique_id)] = {
                    'left': settings.get('left'),
                    'top': settings.get('top'),
                    'scaleX': settings.get('scaleX'),
                    'scaleY': settings.get('scaleY'),
                    'canvasWidth': settings.get('canvasWidth'),
                    'canvasHeight': settings.get('canvasHeight'),
                    'overallScale': settings.get('overallScale')
                }
                
                return web.json_response({
                    "status": "success",
                    "message": "Reference settings updated successfully",
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
