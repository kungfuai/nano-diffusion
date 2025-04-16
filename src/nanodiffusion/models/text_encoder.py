import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer


class TextEncoder(nn.Module):
    def __init__(self, model_name: str, device: str):
        super().__init__()
        self.model_name = model_name
        self.model = CLIPTextModel.from_pretrained(model_name).to(device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.device = device
        # Get the text embedding dimension from the config
        self.text_embed_dim = self.model.config.hidden_size

    def forward(self, text: list[str]) -> torch.Tensor:
        tokens = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(self.device)
        output = self.model(**tokens)
        return output.pooler_output