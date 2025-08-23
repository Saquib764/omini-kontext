from diffusers.pipelines import FluxPipeline
from diffusers.utils import logging
from diffusers.pipelines.flux.pipeline_flux import logger
from torch import Tensor
import numpy as np
from PIL import Image, ImageChops, ImageFilter


def encode_images(pipeline: FluxPipeline, images: Tensor):
    images = pipeline.image_processor.preprocess(images)
    images = images.to(pipeline.device).to(pipeline.dtype)
    images = pipeline.vae.encode(images).latent_dist.sample()
    images = (
        images - pipeline.vae.config.shift_factor
    ) * pipeline.vae.config.scaling_factor
    images_tokens = pipeline._pack_latents(images, *images.shape)
    images_ids = pipeline._prepare_latent_image_ids(
        images.shape[0],
        images.shape[2],
        images.shape[3],
        pipeline.device,
        pipeline.dtype,
    )
    if images_tokens.shape[1] != images_ids.shape[0]:
        images_ids = pipeline._prepare_latent_image_ids(
            images.shape[0],
            images.shape[2] // 2,
            images.shape[3] // 2,
            pipeline.device,
            pipeline.dtype,
        )
    return images_tokens, images_ids


def prepare_text_input(pipeline: FluxPipeline, prompts, max_sequence_length=512):
    # Turn off warnings (CLIP overflow)
    logger.setLevel(logging.ERROR)
    (
        prompt_embeds,
        pooled_prompt_embeds,
        text_ids,
    ) = pipeline.encode_prompt(
        prompt=prompts,
        prompt_2=None,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        device=pipeline.device,
        num_images_per_prompt=1,
        max_sequence_length=max_sequence_length,
        lora_scale=None,
    )
    # Turn on warnings
    logger.setLevel(logging.WARNING)
    return prompt_embeds, pooled_prompt_embeds, text_ids


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
