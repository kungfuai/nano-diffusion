from torch.utils.data import Dataset
import numpy as np
import os
import torch
from tqdm import tqdm


def download_file_with_progress(url, output_path, overwrite=False):
    import requests

    if os.path.exists(output_path) and not overwrite:
        print(f"File {output_path} already exists. Skipping download.")
        return

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    # make sure the parent directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        downloaded_size = 0
        for chunk in tqdm(response.iter_content(chunk_size=1024), total=total_size, unit='B', unit_scale=True):
            if chunk:
                f.write(chunk)
                downloaded_size += len(chunk)


class MJLatentsDataset(Dataset):
    def __init__(self, root_dir='data/raw'):
        self.root_dir = root_dir
        
        url = 'https://huggingface.co/apapiu/small_ldt/resolve/main/mj_latents.npy'
        download_file_with_progress(url, os.path.join(self.root_dir, 'mj_latents.npy'))
        url = 'https://huggingface.co/apapiu/small_ldt/resolve/main/mj_text_emb.npy'
        download_file_with_progress(url, os.path.join(self.root_dir, 'mj_text_emb.npy'))

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