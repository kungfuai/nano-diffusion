from torch.utils.data import Dataset
import numpy as np
import os
import torch

# TODO: add a downloader of the npy files (can download to the cache dir)

class MJLatentsDataset(Dataset):
    def __init__(self, root_dir='data/raw'):
        self.root_dir = root_dir
        self.latents = np.load(os.path.join(root_dir, 'mj_latents.npy'))
        self.texts = np.load(os.path.join(root_dir, 'mj_text_emb.npy'))

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        return {
            'image_emb': self.latents[idx],
            'text_emb': self.texts[idx],
        }


def collate_fn(batch):
    assert "image_emb" in batch[0], f"Data must be a dict that contains 'image_emb'. Got {type(batch[0])}"
    data = {
        'image_emb': torch.stack([torch.from_numpy(np.array(item['image_emb'])) for item in batch]).float(),
    }
    if "text_emb" in batch[0]:
        text_emb = torch.stack([torch.from_numpy(np.array(item["text_emb"])) for item in batch])
        data["text_emb"] = text_emb.reshape(text_emb.shape[0], -1).float()
    return data