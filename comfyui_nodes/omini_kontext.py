import node_helpers
import torch
from einops import rearrange
import comfy.ldm.flux.model
import types
import math
import comfy.conds
import comfy.hooks
import comfy.model_base
import comfy.utils
import copy

try:
    from nunchaku.caching.utils import cache_context, create_cache_context
    from nunchaku.lora.flux.compose import compose_lora
    from nunchaku.utils import load_state_dict_in_safetensors
except:
    pass

def extra_conds(self, **kwargs):
    out = self._extra_conds(**kwargs)
    
    omini_latents = kwargs.get("omini_latents", None)
    if omini_latents is not None:
        latents = []
        deltas = [] 
        for cond in omini_latents:
            lat = cond["latent"]
            delta = cond["delta"]
            latents.append(self.process_latent_in(lat))
            deltas.append(torch.tensor([[[delta]]], device=lat.device))
        out['omini_latents'] = comfy.conds.CONDList(latents)
        out['omini_latents_deltas'] = comfy.conds.CONDList(deltas)
    return out

def nunchaku_extra_conds(self, **kwargs):
    out = self._nunchaku_extra_conds(**kwargs)
    
    omini_latents = kwargs.get("omini_latents", None)
    if omini_latents is not None:
        latents = []
        deltas = [] 
        for cond in omini_latents:
            lat = cond["latent"]
            delta = cond["delta"]
            latents.append(self.process_latent_in(lat))
            deltas.append(torch.tensor([[[delta]]], device=lat.device))
        out['omini_latents'] = comfy.conds.CONDList(latents)
        out['omini_latents_deltas'] = comfy.conds.CONDList(deltas)
    return out

def extra_conds_shapes(self, **kwargs):
    out = self._extra_conds_shapes(**kwargs)
    out = {}
    omini_latents = kwargs.get("omini_latents", None)
    omini_latents_deltas = kwargs.get("omini_latents_deltas", None)
    if omini_latents is not None:
        out['omini_latents'] = list([1, 16, sum(map(lambda a: math.prod(a.size()), omini_latents)) // 16])
    if omini_latents_deltas is not None:
        out['omini_latents_deltas'] = list([1, 1, sum(map(lambda a: math.prod(a.size()), omini_latents_deltas))])
    return out


def new_forward(self, x, timestep, context, y=None, guidance=None, ref_latents=None, control=None, transformer_options={}, omini_latents=None, omini_latents_deltas=None, **kwargs):
    bs, c, h_orig, w_orig = x.shape
    patch_size = self.patch_size

    h_len = ((h_orig + (patch_size // 2)) // patch_size)
    w_len = ((w_orig + (patch_size // 2)) // patch_size)
    img, img_ids = self.process_img(x)
    img_tokens = img.shape[1]
    if ref_latents is not None:
        h = 0
        w = 0
        for ref in ref_latents:
            h_offset = 0
            w_offset = 0
            if ref.shape[-2] + h > ref.shape[-1] + w:
                w_offset = w
            else:
                h_offset = h

            kontext, kontext_ids = self.process_img(ref, index=1, h_offset=h_offset, w_offset=w_offset)
            img = torch.cat([img, kontext], dim=1)
            img_ids = torch.cat([img_ids, kontext_ids], dim=1)
            h = max(h, ref.shape[-2] + h_offset)
            w = max(w, ref.shape[-1] + w_offset)
    
    if omini_latents is not None:
        for lat, delta in zip(omini_latents, omini_latents_deltas):
            i_offset, h_offset, w_offset = delta[0,0,0].tolist()
            kontext, kontext_ids = self.process_img(lat, index=1+i_offset, h_offset=h_offset * self.patch_size, w_offset=w_offset * self.patch_size)
            img = torch.cat([img, kontext], dim=1)
            img_ids = torch.cat([img_ids, kontext_ids], dim=1)

    txt_ids = torch.zeros((bs, context.shape[1], 3), device=x.device, dtype=x.dtype)
    out = self.forward_orig(img, img_ids, context, txt_ids, timestep, y, guidance, control, transformer_options, attn_mask=kwargs.get("attention_mask", None))
    out = out[:, :img_tokens]
    return rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=2, pw=2)[:,:,:h_orig,:w_orig]

def is_flux_model(model):
    if isinstance(model, comfy.ldm.flux.model.Flux):
        return True
    return False

class OminiKontextModelPatch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL", ),
                             }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_patch"

    CATEGORY = "model_patches/unet"

    def apply_patch(self, model):
        new_model = model.clone()
        if is_flux_model(new_model.get_model_object('diffusion_model')):
            diffusion_model = new_model.get_model_object('diffusion_model')
            # Replace the forward method with the new one type 
            diffusion_model.forward = types.MethodType(new_forward, diffusion_model)

            # Now backup and replace the extra_conds and extra_conds_shapes methods
            new_model.model._extra_conds = new_model.model.extra_conds
            # new_model.model._extra_conds_shapes = new_model.model.extra_conds_shapes
            new_model.model.extra_conds = types.MethodType(extra_conds, new_model.model)
            # new_model.model.extra_conds_shapes = types.MethodType(extra_conds_shapes, new_model.model)
        return (new_model,)


class OminiKontextConditioning:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", ),
                             "latent": ("LATENT", ),
                             "delta_0": ("INT", {"default": 0, "min": -100, "max": 100}),
                             "delta_1": ("INT", {"default": 0, "min": -200, "max": 200}),
                             "delta_2": ("INT", {"default": 0, "min": -200, "max": 200})
                            },
               }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "append"

    CATEGORY = "advanced/conditioning/edit_models"
    DESCRIPTION = "This node sets the reference latent for Flux Kontext model. By default, the model doesn't support two images as input, so this model requires a LoRA trained with omini-kontext framework."

    def append(self, conditioning, latent=None, delta_0=0, delta_1=0, delta_2=0):
        if latent is not None:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"omini_latents": [{"latent": latent["samples"], "delta": [delta_0, delta_1, delta_2]}]}, append=True)
        return (conditioning, )


def new_nunchaku_forward(self, x, timestep, context, y=None, guidance=None, control=None, transformer_options={}, **kwargs):

    if isinstance(timestep, torch.Tensor):
        if timestep.numel() == 1:
            timestep_float = timestep.item()
        else:
            timestep_float = timestep.flatten()[0].item()
    else:
        assert isinstance(timestep, float)
        timestep_float = timestep

    model = self.model

    bs, c, h_orig, w_orig = x.shape
    patch_size = self.config.get("patch_size", 2)
    h_len = (h_orig + (patch_size // 2)) // patch_size
    w_len = (w_orig + (patch_size // 2)) // patch_size

    img, img_ids = self.process_img(x)
    img_tokens = img.shape[1]

    ref_latents = kwargs.get("ref_latents")
    if ref_latents is not None:
        h = 0
        w = 0
        for ref in ref_latents:
            h_offset = 0
            w_offset = 0
            if ref.shape[-2] + h > ref.shape[-1] + w:
                w_offset = w
            else:
                h_offset = h

            kontext, kontext_ids = self.process_img(ref, index=1, h_offset=h_offset, w_offset=w_offset)
            img = torch.cat([img, kontext], dim=1)
            img_ids = torch.cat([img_ids, kontext_ids], dim=1)
            h = max(h, ref.shape[-2] + h_offset)
            w = max(w, ref.shape[-1] + w_offset)
    
    omini_latents = kwargs.get("omini_latents")
    omini_latents_deltas = kwargs.get("omini_latents_deltas")
    if omini_latents is not None:
        for lat, delta in zip(omini_latents, omini_latents_deltas):
            i_offset, h_offset, w_offset = delta[0,0,0].tolist()
            kontext, kontext_ids = self.process_img(lat, index=1+i_offset, h_offset=h_offset * patch_size, w_offset=w_offset * patch_size)
            img = torch.cat([img, kontext], dim=1)
            img_ids = torch.cat([img_ids, kontext_ids], dim=1)

    txt_ids = torch.zeros((bs, context.shape[1], 3), device=x.device, dtype=x.dtype)


    # load and compose LoRA
    if self.loras != model.comfy_lora_meta_list:
        lora_to_be_composed = []
        for _ in range(max(0, len(model.comfy_lora_meta_list) - len(self.loras))):
            model.comfy_lora_meta_list.pop()
            model.comfy_lora_sd_list.pop()
        for i in range(len(self.loras)):
            meta = self.loras[i]
            if i >= len(model.comfy_lora_meta_list):
                sd = load_state_dict_in_safetensors(meta[0])
                model.comfy_lora_meta_list.append(meta)
                model.comfy_lora_sd_list.append(sd)
            elif model.comfy_lora_meta_list[i] != meta:
                if meta[0] != model.comfy_lora_meta_list[i][0]:
                    sd = load_state_dict_in_safetensors(meta[0])
                    model.comfy_lora_sd_list[i] = sd
                model.comfy_lora_meta_list[i] = meta
            lora_to_be_composed.append(({k: v for k, v in model.comfy_lora_sd_list[i].items()}, meta[1]))

        composed_lora = compose_lora(lora_to_be_composed)

        if len(composed_lora) == 0:
            model.reset_lora()
        else:
            if "x_embedder.lora_A.weight" in composed_lora:
                new_in_channels = composed_lora["x_embedder.lora_A.weight"].shape[1]
                current_in_channels = model.x_embedder.in_features
                if new_in_channels < current_in_channels:
                    model.reset_x_embedder()
            model.update_lora_params(composed_lora)

    controlnet_block_samples = None if control is None else [y.to(x.dtype) for y in control["input"]]
    controlnet_single_block_samples = None if control is None else [y.to(x.dtype) for y in control["output"]]

    if self.pulid_pipeline is not None:
        self.model.transformer_blocks[0].pulid_ca = self.pulid_pipeline.pulid_ca

    if getattr(model, "residual_diff_threshold_multi", 0) != 0 or getattr(model, "_is_cached", False):
        # A more robust caching strategy
        cache_invalid = False

        # Check if timestamps have changed or are out of valid range
        if self._prev_timestep is None:
            cache_invalid = True
        elif self._prev_timestep < timestep_float + 1e-5:  # allow a small tolerance to reuse the cache
            cache_invalid = True

        if cache_invalid:
            self._cache_context = create_cache_context()

        # Update the previous timestamp
        self._prev_timestep = timestep_float
        with cache_context(self._cache_context):
            if self.customized_forward is None:
                out = model(
                    hidden_states=img,
                    encoder_hidden_states=context,
                    pooled_projections=y,
                    timestep=timestep,
                    img_ids=img_ids,
                    txt_ids=txt_ids,
                    guidance=guidance if self.config["guidance_embed"] else None,
                    controlnet_block_samples=controlnet_block_samples,
                    controlnet_single_block_samples=controlnet_single_block_samples,
                ).sample
            else:
                out = self.customized_forward(
                    model,
                    hidden_states=img,
                    encoder_hidden_states=context,
                    pooled_projections=y,
                    timestep=timestep,
                    img_ids=img_ids,
                    txt_ids=txt_ids,
                    guidance=guidance if self.config["guidance_embed"] else None,
                    controlnet_block_samples=controlnet_block_samples,
                    controlnet_single_block_samples=controlnet_single_block_samples,
                    **self.forward_kwargs,
                ).sample
    else:
        if self.customized_forward is None:
            out = model(
                hidden_states=img,
                encoder_hidden_states=context,
                pooled_projections=y,
                timestep=timestep,
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=guidance if self.config["guidance_embed"] else None,
                controlnet_block_samples=controlnet_block_samples,
                controlnet_single_block_samples=controlnet_single_block_samples,
            ).sample
        else:
            out = self.customized_forward(
                model,
                hidden_states=img,
                encoder_hidden_states=context,
                pooled_projections=y,
                timestep=timestep,
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=guidance if self.config["guidance_embed"] else None,
                controlnet_block_samples=controlnet_block_samples,
                controlnet_single_block_samples=controlnet_single_block_samples,
                **self.forward_kwargs,
            ).sample
    if self.pulid_pipeline is not None:
        self.model.transformer_blocks[0].pulid_ca = None

    out = out[:, :img_tokens]
    out = rearrange(
        out,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=h_len,
        w=w_len,
        ph=patch_size,
        pw=patch_size,
    )
    out = out[:, :, :h_orig, :w_orig]

    self._prev_timestep = timestep_float
    return out
    

class NunchakuOminiKontextPatch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL", ),
                             }}
    RETURN_TYPES = ("MODEL",)
    OUTPUT_TOOLTIPS = ("The modified diffusion model.",)
    FUNCTION = "load_omini_kontext"

    CATEGORY = "model_patches/unet"

    def load_omini_kontext(self, model):
        new_model = model.clone()
        diffusion_model = new_model.get_model_object('diffusion_model')
        # Replace the forward method with the new one type 
        diffusion_model.forward = types.MethodType(new_nunchaku_forward, diffusion_model)

        # Now backup and replace the extra_conds and extra_conds_shapes methods
        # Use unique backup name to avoid conflicts with other patches
        new_model.model._nunchaku_extra_conds = new_model.model.extra_conds
        # new_model.model._extra_conds_shapes = new_model.model.extra_conds_shapes
        new_model.model.extra_conds = types.MethodType(nunchaku_extra_conds, new_model.model)
        # new_model.model.extra_conds_shapes = types.MethodType(extra_conds_shapes, new_model.model)


        return (new_model,)