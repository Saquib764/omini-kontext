import node_helpers

class OminiKontextModelPatch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL", ),
                             }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_patch"

    CATEGORY = "advanced/conditioning/edit_models"

    def apply_patch(self, model):
        print(model)
        print(model.model)
        print(model.model.diffusion_model)
        return (model,)


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
            conditioning = node_helpers.conditioning_set_values(conditioning, {"omini_latents": [{"latent":latent["samples"]}], "delta": [delta_0, delta_1, delta_2]}, append=True)
        return (conditioning, )


