import torch
from src.pipeline_flux_omini_kontext import FluxOminiKontextPipeline
from diffusers.utils import load_image

pipe = FluxOminiKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16
)
pipe.to("cuda")

image = load_image(
    "https://firebasestorage.googleapis.com/v0/b/saquib-sh.appspot.com/o/thefluxtrain%2Fv02%2Feditor%2F35ae2291-a199-4a2e-9742-f9b0c43c3152%2Frgj44h.png?alt=media"
).convert("RGB")

reference = load_image(
    "https://firebasestorage.googleapis.com/v0/b/saquib-sh.appspot.com/o/thefluxtrain%2Fv02%2Feditor%2F35ae2291-a199-4a2e-9742-f9b0c43c3152%2Fyv6onr.png?alt=media"
).convert("RGB")
prompt = "add a cat to the image"
result = pipe(
    image=image,
    reference=reference,
    prompt=prompt,
    guidance_scale=2.5,
    generator=torch.Generator().manual_seed(42),
).images[0]
result.save("output.png")