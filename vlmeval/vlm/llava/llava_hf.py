import os
import os.path as osp
import string
import sys
import warnings

import pandas as pd
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    Blip2Processor,
    Blip2ForConditionalGeneration,
    InstructBlipProcessor,
    InstructBlipForConditionalGeneration,
)

from ..base import BaseModel
from ...smp import cn_string, get_cache_path
from ...dataset import DATASET_TYPE

class LLaVA_HF(BaseModel):
    
    def __init__(self,
                 model_path='llava-hf/llava-1.5-7b-hf',
                 **kwargs):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_path, device_map="auto", torch_dtype=torch.bfloat16
        )
    
    def generate_inner(self, message, dataset=None):
        texts, images = [], []
        for m in message:
            if m['type'] == 'image':
                images.append(Image.open(m['value']).convert('RGB'))
            elif m['type'] == 'text':
                texts.append(m['value'].strip())
            else:
                raise ValueError(f"Unknown message type: {m['type']}")
        if len(images) > 1:
            pass
        assert len(images) == 1
        text = ' '.join(texts)
        prompts = [f"USER: <image>\n{text}\nASSISTANT:"]
        inputs = self.processor(
            text=prompts, images=images, return_tensors="pt"
        ).to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=512)
        response = self.processor.batch_decode(outputs, skip_special_tokens=True)
        
        return response[0].split("ASSISTANT:")[1].strip()