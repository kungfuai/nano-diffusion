from dataclasses import dataclass
from typing import Dict, Optional
import torch
import numpy as np


@dataclass
class MiniBatch:
    x: torch.Tensor  # latent or pixel space input
    text_emb: Optional[torch.Tensor] = None
    cond_emb_dict: Optional[Dict[str, torch.Tensor]] = None
    
    def to(self, device: torch.device) -> 'MiniBatch':
        """Move all tensors to device"""
        x = self.x.to(device)
        text_emb = self.text_emb.to(device) if self.text_emb is not None else None
        cond_emb_dict = {
            k: v.to(device) for k, v in self.cond_emb_dict.items() if v is not None
        } if self.cond_emb_dict else None
        return MiniBatch(x=x, text_emb=text_emb, cond_emb_dict=cond_emb_dict)
    
    @property
    def device(self) -> torch.device:
        """Return the device of the first tensor in the batch."""
        return self.x.device
    
    @property
    def num_examples(self) -> int:
        """Return the number of examples in the batch."""
        return self.x.shape[0]
    
    @property
    def has_conditional_embeddings(self) -> bool:
        """Return True if the batch has conditional embeddings."""
        return self.cond_emb_dict is not None
    
    @staticmethod   
    def from_dataloader_batch(batch) -> "MiniBatch":
        """Convert raw batch dict to MiniBatch object.

        Args:
            batch: A dictionary containing the batch data. Typically a batch from a DataLoader.

        Returns:
            A MiniBatch object containing the parsed batch data.
        """
        parsed_batch = parse_batch(batch)
        return parsed_batch
        


def parse_batch(batch) -> MiniBatch:
    """Convert raw batch dict to MiniBatch object"""
    if isinstance(batch, tuple) or isinstance(batch, list):
        x = batch[0]
        text_emb = None
        cond_emb_dict = None
        if len(batch) == 2:
            if isinstance(batch[1], dict):
                text_emb = batch[1].get("text_emb")
            else:
                text_emb = batch[1]
            # print(text_emb[0], type(text_emb))
            text_emb = text_emb.float().reshape(text_emb.shape[0], -1)
            cond_emb_dict = {"text_emb": text_emb}

    elif isinstance(batch, dict):
        if "image_emb" in batch:
            # This means the image embeddings in the latent space are already provided.
            x = batch["image_emb"].float()
        else:
            x = batch["image"].float()
        text_emb = batch.get("text_emb")
        if text_emb is not None:
            text_emb = text_emb.float().reshape(text_emb.shape[0], -1)
            
        # Optional: collect conditional embeddings
        cond_emb_dict = {}
        for key in batch:
            if key.endswith("_emb") and key not in ["image_emb"]:
                cond_emb_dict[key] = batch[key].float()
    
    elif isinstance(batch, torch.Tensor):
        x = batch.float()
        text_emb = None
        cond_emb_dict = None

    else:
        raise ValueError(f"Unsupported batch type: {type(batch)}")

    return MiniBatch(
        x=x,
        text_emb=text_emb,
        cond_emb_dict=cond_emb_dict if cond_emb_dict else None
    )


def collate_fn_for_latents(batch):
    if isinstance(batch[0], dict):
        assert "image_emb" in batch[0], f"Data must be a list of dicts that contains 'image_emb'. Got {type(batch[0])}. {batch}"
    elif isinstance(batch[0], tuple):
        # Expecting a tuple (image_emb, text_emb)
        batch = [
            {"image_emb": x[0], "text_emb": x[1]} for x in batch
        ]
    # print(batch[0]['image_emb'])
    # assert np.array(batch[0]['image_emb']).shape == (4, 8, 8), f"Image emb shape is {np.array(batch[0]['image_emb']).shape}. Expected (4, 8, 8)."
    data = {
        'image_emb': torch.stack([torch.from_numpy(np.array(item['image_emb'])) for item in batch]),
    }
    if "text_emb" in batch[0]:
        data["text_emb"] = torch.stack([torch.from_numpy(np.array(item["text_emb"])) for item in batch])
    return data
