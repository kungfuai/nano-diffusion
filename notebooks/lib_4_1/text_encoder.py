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
        # return output.last_hidden_state
        return output.pooler_output

if __name__ == "__main__":
    choices = [
        "openai/clip-vit-large-patch14",
        "openai/clip-vit-base-patch32",
        "zer0int/CLIP-GmP-ViT-L-14",
        "google/paligemma2-3b-pt-224",
        "Salesforce/blip2-opt-2.7b",
    ]
    name = choices[0]
    text_encoder = TextEncoder(name, "cuda:0")
    text = "a photo of a cat"
    text2 = "a photo of a big wolf"
    embeddings = text_encoder([text, text2])
    print(embeddings.shape)
    print("text_embed_dim:", text_encoder.text_embed_dim)
