from .comfyui_nodes.omini_kontext import OminiKontextConditioning, OminiKontextModelPatch, NunchakuOminiKontextPatch
from .comfyui_nodes.omini_qwen_image import OminiQwenImageEditModelPatch
from .comfyui_nodes.omini_kontext_editor import NODE_CLASS_MAPPINGS as OMINI_KONTEXT_EDITOR_NODES, NODE_DISPLAY_NAME_MAPPINGS as OMINI_KONTEXT_EDITOR_NODES_NAMES
import os

WEB_DIRECTORY = os.path.join(os.path.dirname(__file__), "comfyui_nodes", "js")

NODE_CLASS_MAPPINGS = {
    "OminiKontextConditioning": OminiKontextConditioning,
    "OminiKontextModelPatch": OminiKontextModelPatch,
    "NunchakuOminiKontextPatch": NunchakuOminiKontextPatch,
    "OminiQwenImageEditModelPatch": OminiQwenImageEditModelPatch,
    **OMINI_KONTEXT_EDITOR_NODES
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "OminiKontextConditioning": "Omini Kontext Conditioning",
    "OminiKontextModelPatch": "Omini Kontext Model Patch",
    "NunchakuOminiKontextPatch": "Nunchaku Omini Kontext Patch",
    "OminiQwenImageEditModelPatch": "Omini Qwen Image Edit Model Patch",
    **OMINI_KONTEXT_EDITOR_NODES_NAMES
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]