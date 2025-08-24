import torch
import types
import comfy.ldm.flux.model
from .omini_kontext import extra_conds


def new_forward(
    self,
    x,
    timesteps,
    context,
    attention_mask=None,
    guidance: torch.Tensor = None,
    ref_latents=None,
    transformer_options={},
    control=None,
    omini_latents=None,
    omini_latents_deltas=None,
    **kwargs
):
    timestep = timesteps
    encoder_hidden_states = context
    encoder_hidden_states_mask = attention_mask

    hidden_states, img_ids, orig_shape = self.process_img(x)
    num_embeds = hidden_states.shape[1]

    if ref_latents is not None:
        h = 0
        w = 0
        index = 0
        index_ref_method = kwargs.get("ref_latents_method", "index") == "index"
        for ref in ref_latents:
            if index_ref_method:
                index += 1
                h_offset = 0
                w_offset = 0
            else:
                index = 1
                h_offset = 0
                w_offset = 0
                if ref.shape[-2] + h > ref.shape[-1] + w:
                    w_offset = w
                else:
                    h_offset = h
                h = max(h, ref.shape[-2] + h_offset)
                w = max(w, ref.shape[-1] + w_offset)

            kontext, kontext_ids, _ = self.process_img(ref, index=index, h_offset=h_offset, w_offset=w_offset)
            hidden_states = torch.cat([hidden_states, kontext], dim=1)
            img_ids = torch.cat([img_ids, kontext_ids], dim=1)
    
    if omini_latents is not None:
        for lat, delta in zip(omini_latents, omini_latents_deltas):
            i_offset, h_offset, w_offset = delta[0,0,0].tolist()
            kontext, kontext_ids, _ = self.process_img(lat, index=1+i_offset, h_offset=h_offset * self.patch_size, w_offset=w_offset * self.patch_size)
            hidden_states = torch.cat([hidden_states, kontext], dim=1)
            img_ids = torch.cat([img_ids, kontext_ids], dim=1)

    txt_start = round(max(((x.shape[-1] + (self.patch_size // 2)) // self.patch_size) // 2, ((x.shape[-2] + (self.patch_size // 2)) // self.patch_size) // 2))
    txt_ids = torch.arange(txt_start, txt_start + context.shape[1], device=x.device).reshape(1, -1, 1).repeat(x.shape[0], 1, 3)
    ids = torch.cat((txt_ids, img_ids), dim=1)
    image_rotary_emb = self.pe_embedder(ids).squeeze(1).unsqueeze(2).to(x.dtype)
    del ids, txt_ids, img_ids

    hidden_states = self.img_in(hidden_states)
    encoder_hidden_states = self.txt_norm(encoder_hidden_states)
    encoder_hidden_states = self.txt_in(encoder_hidden_states)

    if guidance is not None:
        guidance = guidance * 1000

    temb = (
        self.time_text_embed(timestep, hidden_states)
        if guidance is None
        else self.time_text_embed(timestep, guidance, hidden_states)
    )

    patches_replace = transformer_options.get("patches_replace", {})
    patches = transformer_options.get("patches", {})
    blocks_replace = patches_replace.get("dit", {})

    for i, block in enumerate(self.transformer_blocks):
        if ("double_block", i) in blocks_replace:
            def block_wrap(args):
                out = {}
                out["txt"], out["img"] = block(hidden_states=args["img"], encoder_hidden_states=args["txt"], encoder_hidden_states_mask=encoder_hidden_states_mask, temb=args["vec"], image_rotary_emb=args["pe"])
                return out
            out = blocks_replace[("double_block", i)]({"img": hidden_states, "txt": encoder_hidden_states, "vec": temb, "pe": image_rotary_emb}, {"original_block": block_wrap})
            hidden_states = out["img"]
            encoder_hidden_states = out["txt"]
        else:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )

        if "double_block" in patches:
            for p in patches["double_block"]:
                out = p({"img": hidden_states, "txt": encoder_hidden_states, "x": x, "block_index": i})
                hidden_states = out["img"]
                encoder_hidden_states = out["txt"]

        if control is not None: # Controlnet
            control_i = control.get("input")
            if i < len(control_i):
                add = control_i[i]
                if add is not None:
                    hidden_states += add

    hidden_states = self.norm_out(hidden_states, temb)
    hidden_states = self.proj_out(hidden_states)

    hidden_states = hidden_states[:, :num_embeds].view(orig_shape[0], orig_shape[-2] // 2, orig_shape[-1] // 2, orig_shape[1], 2, 2)
    hidden_states = hidden_states.permute(0, 3, 1, 4, 2, 5)
    return hidden_states.reshape(orig_shape)[:, :, :, :x.shape[-2], :x.shape[-1]]

class OminiQwenImageEditModelPatch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL", ),
                             }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_patch"

    CATEGORY = "model_patches/unet"

    def apply_patch(self, model):
        new_model = model.clone()
        diffusion_model = new_model.get_model_object('diffusion_model')
        # Replace the forward method with the new one type 
        diffusion_model._forward = types.MethodType(new_forward, diffusion_model)

        # Now backup and replace the extra_conds and extra_conds_shapes methods
        new_model.model._extra_conds = model.model.extra_conds
        # new_model.model._extra_conds_shapes = new_model.model.extra_conds_shapes
        new_model.model.extra_conds = types.MethodType(extra_conds, new_model.model)
        # new_model.model.extra_conds_shapes = types.MethodType(extra_conds_shapes, new_model.model)
        return (new_model,)

