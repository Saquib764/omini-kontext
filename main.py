import torch
from PIL import Image
from src.pipeline_flux_omini_kontext import FluxOminiKontextPipeline
from diffusers.utils import load_image

pipe = FluxOminiKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16
)
pipe.to("cuda")

image = Image.open("assets/boy_scene_small.png")

reference = Image.open("assets/boy_reference_256.png")


prompt = "add a cat to the image"
result = pipe(
    image=image,
    reference=reference,
    prompt=prompt,
    guidance_scale=2.5,
    generator=torch.Generator().manual_seed(42),
).images[0]
result.save("output.png")