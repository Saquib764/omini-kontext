# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import time
from cog import BasePredictor, Input, Path, Secret
import torch
from PIL import Image, ImageChops
from src.pipeline_qwen_omini_image_edit import QwenOminiImageEditPipeline
import random
import json

from utils import (
    ensure_hf_login,
    optimise_image_condition
)

LoRA_MODELS = {
    "none": {
        "lora_path": None,
        "weight_name": None,
    },
    "spatial_character_insertion": {
        "lora_path": "saquiboye/omini-kontext",
        "weight_name": "qwen/character_spatial_1000.safetensors",
    },
    "character_insertion": {
        "lora_path": "saquiboye/omini-kontext",
        "weight_name": "qwen/character_1000.safetensors",
    },
    # "product_insertion": {
    #     "lora_path": "saquiboye/omini-kontext",
    #     "weight_name": "product_2000.safetensors",
    # }
}

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        ensure_hf_login()
        self.pipe = QwenOminiImageEditPipeline.from_pretrained(
            "Qwen/Qwen-Image-Edit", torch_dtype=torch.bfloat16
        ).to("cuda")

    def predict(
        self,
        image: Path = Input(
            description="Image to insert character into", default=None
        ),
        reference_image: Path = Input(
            description="Reference image", default=None
        ),
        task: str = Input(
            description="Task",
            choices=["character_insertion", "spatial_character_insertion", "product_insertion", "none", 'custom'],
            default="character_insertion"
        ),
        delta: str = Input(
            description="Reference delta", default="[1, 0, 0]"
        ),
        prompt: str = Input(
            description="Input prompt.",
            default="Add character to the scene",
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=150, default=20
        ),
        lora_strength: float = Input(
            description="LoRA strength", ge=0.0, le=1.0, default=0.6
        ),
        guidance_scale: float = Input(
            description="Text guidance scale", ge=0.0, le=10.0, default=4.5
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        lora_path: str = Input(
            description="HF path to the LoRA weights, if custom task is choosen", default=None
        ),
        lora_weight_name: str = Input(
            description="Weight name of the LoRA weights, if custom task is choosen", default=None
        )
    ) -> Path:
        """Run a single prediction on the model"""
        if image is None:
            raise ValueError("'image' must be provided.")
        delta = json.loads(delta)

        self.pipe.unload_lora_weights()
        lora = None
        if task == 'custom':
            lora = {
                "lora_path": lora_path,
                "weight_name": lora_weight_name
            }
        elif task != 'none':
            lora = LoRA_MODELS[task]

        if lora is not None:
            self.pipe.load_lora_weights(
                lora["lora_path"],
                weight_name=lora["weight_name"],
                adapter_name="reference"
            )
            self.pipe.set_adapters("reference", adapter_weights=lora_strength)

        # Setup generation parameters
        seed = random.randint(0, 65535) if seed is None else seed
        print(f"Using seed: {seed}")
        generator = torch.Generator("cuda").manual_seed(seed)

        image = Image.open(image).convert("RGB")
        has_reference = reference_image is not None
        if has_reference:
            reference_image = Image.open(reference_image).convert("RGB")

        width, height = image.size

        MAX_SIZE = 1536
        # Compute new width and height, maintaining aspect ratio, with max side = MAX_SIZE
        if max(width, height) > MAX_SIZE:
            if width >= height:
                new_width = MAX_SIZE
                new_height = int(height * (MAX_SIZE / width))
            else:
                new_height = MAX_SIZE
                new_width = int(width * (MAX_SIZE / height))
            width = int((new_width//16) * 16)
            height = int((new_height//16) * 16)
            image = image.resize((width, height), Image.LANCZOS)
            if has_reference:
                reference_image = reference_image.resize((width, height), Image.LANCZOS)
        
        try:
            print("has_reference: ", has_reference)
            print("reference_image: ", reference_image)
            print("delta: ", delta)
            if has_reference:
                optimised_reference, new_reference_delta = optimise_image_condition(reference_image, delta)
            result_img = self.pipe(
                prompt=prompt,
                image=image,
                reference=optimised_reference if has_reference else None,
                reference_delta=new_reference_delta if has_reference else None,
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
                generator=generator,
                # _auto_resize=False,
                # max_area=width*height,
                guidance_scale=guidance_scale
            ).images[0]

        finally:
            # Cleanup
            self.pipe.unload_lora_weights()
        # Resize back to the original size
        result_img = result_img.resize((width, height))
        print("result_img: ", result_img)

        out_path = "/tmp/out.png"
        result_img.save(out_path)
        return Path(out_path)
            