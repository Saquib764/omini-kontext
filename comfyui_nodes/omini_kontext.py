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


def extra_conds(self, **kwargs):
    out = self._extra_conds(**kwargs)
    
    omini_latents = kwargs.get("omini_latents", None)
    if omini_latents is not None:
        latents = []
        deltas = [] 
        for cond in omini_latents:
            lat = cond["latent"]
            delta = cond["delta"]
            print("lat", lat.shape)
            print("delta", torch.tensor([[[delta]]], device=lat.device).shape)
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
    print("omini_latents_deltas", omini_latents_deltas)
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
            kontext, kontext_ids = self.process_img(lat, index=1+i_offset, h_offset=h_offset, w_offset=w_offset)
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
                             "delta_1": ("INT", {"default": 0, "min": -100, "max": 100}),
                             "delta_2": ("INT", {"default": 0, "min": -100, "max": 100})
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


