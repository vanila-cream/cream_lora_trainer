"""
Cream LoRA Trainer - ComfyUI Custom Node
Trains SDXL LoRAs from dataset folders using kohya sd-scripts.
Auto-captions images using WD tagger models.
"""
from .cream_lora_trainer import CreamLoraTrainer
from .cream_auto_captioner import CreamAutoCaptioner
from .cream_common_tag_extractor import CreamCommonTagExtractor

NODE_CLASS_MAPPINGS = {
    "CreamLoraTrainer": CreamLoraTrainer,
    "CreamAutoCaptioner": CreamAutoCaptioner,
    "CreamCommonTagExtractor": CreamCommonTagExtractor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CreamLoraTrainer": "Cream LoRA Trainer",
    "CreamAutoCaptioner": "Cream Auto Captioner",
    "CreamCommonTagExtractor": "Cream Common Tag Extractor",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
