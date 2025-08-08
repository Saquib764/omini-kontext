# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import time
from cog import BasePredictor, Input, Path, Secret
import torch
from PIL import Image, ImageChops
from src.pipeline_flux_omini_kontext import FluxOminiKontextPipeline
import random
import json

from utils import (
    ensure_hf_login,
    optimise_image_condition
)

LoRA_MODELS = {
    "character_insertion": {
        "lora_path": "saquiboye/omini-kontext-character",
        "weight_name": "spatial-character-test.safetensors",
    }
}

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        ensure_hf_login()
        self.pipe = FluxOminiKontextPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16
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
            choices=["character_insertion", 'custom'],
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
        if image is None or reference_image is None:
            raise ValueError("Both 'image' and 'reference_image' must be provided.")
        delta = json.loads(delta)

        self.pipe.unload_lora_weights()

        if task == 'custom':
            lora = {
                "lora_path": lora_path,
                "weight_name": lora_weight_name
            }
        else:
            lora = LoRA_MODELS[task]

        self.pipe.load_lora_weights(
            lora["lora_path"],
            weight_name=lora["weight_name"],
            adapter_name="reference"
        )

        # Setup generation parameters
        seed = random.randint(0, 65535) if seed is None else seed
        print(f"Using seed: {seed}")
        generator = torch.Generator("cuda").manual_seed(seed)

        image = Image.open(image).convert("RGB")
        reference_image = Image.open(reference_image).convert("RGB")

        width, height = image.size

        MAX_SIZE = 1024
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
            reference_image = reference_image.resize((width, height), Image.LANCZOS)
        
        try:
            optimised_reference, new_reference_delta = optimise_image_condition(reference_image, delta)
            result_img = self.pipe(
                prompt=prompt,
                image=image,
                reference=optimised_reference,
                reference_delta=new_reference_delta,
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
                generator=generator,
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
            