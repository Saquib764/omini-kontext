from .comfyui_nodes.omini_kontext import OminiKontextConditioning, OminiKontextModelPatch, NunchakuOminiKontextPatch

NODE_CLASS_MAPPINGS = {
    "OminiKontextConditioning": OminiKontextConditioning,
    "OminiKontextModelPatch": OminiKontextModelPatch,
    "NunchakuOminiKontextPatch": NunchakuOminiKontextPatch,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "OminiKontextConditioning": "Omini Kontext Conditioning",
    "OminiKontextModelPatch": "Omini Kontext Model Patch",
    "NunchakuOminiKontextPatch": "Nunchaku Omini Kontext Patch",
}