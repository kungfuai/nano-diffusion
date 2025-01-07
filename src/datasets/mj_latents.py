from torch.utils.data import Dataset
import numpy as np
import os


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
