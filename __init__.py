from .comfyui_nodes.omini_kontext import OminiKontextConditioning, OminiKontextModelPatch

NODE_CLASS_MAPPINGS = {
    "OminiKontextConditioning": OminiKontextConditioning,
    "OminiKontextModelPatch": OminiKontextModelPatch,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "OminiKontextConditioning": "Omini Kontext Conditioning",
    "OminiKontextModelPatch": "Omini Kontext Model Patch",
}