"""
Image Latents Dataset

This dataset is used to store image latents. Several metadata fields are necessary to make the process reproducible:

- The tokenizer (VAE) used to encode the images into latents. Typically this is a huggingface model name.
- The source image size.
- The shape of the latents.
- The value scaling factor.
- Any other configuration of the tokenizer and the image dataset.
"""

import numpy as np
from tqdm import tqdm
from datasets import load_dataset, Features, Value, Array2D, Array3D, Image
import torch
from torch.utils.data import Dataset
from diffusers import AutoencoderKL
from torchvision.transforms import ToTensor, Resize, Compose, Normalize
from src.utils.data_utils import SquarePad


class ImageLatentsDataset(Dataset):
    """
    Dataset of image latents. This is used to convert a torch image dataset into a latents dataset.

    It includes a `from_image_dataset` method to create a latents dataset from an image dataset.

    Assumptions:
    - The image dataset is a huggingface dataset.
    - The image dataset has an `image` field, and optionally a `text` field (text column can be configured).
    """
    

    def __init__(self,
                 src_dataset_name: str,
                 tokenizer_name: str,
                 image_size: int,
                 split: str = 'train',
                 clip_model_name: str = "ViT-L/14",
                 image_column: str = 'image',
                 text_column: str = 'text',
                 include_original_image: bool = False,
                 device: str = 'cuda:0',
                 limit: int = None,
                 **kwargs):
        self.src_dataset = load_dataset(src_dataset_name)[split]
        self.image_size = image_size
        self.device = device
        self.image_column = image_column
        self.text_column = text_column
        self.include_original_image = include_original_image
        self.limit = limit

        # Initialize models
        self.vae = AutoencoderKL.from_pretrained(tokenizer_name, torch_dtype=torch.float32).to('cuda')

        import clip

        self.clip_model, _ = clip.load(clip_model_name)
        self.clip_model = self.clip_model.to(device)
        print(f"Loaded CLIP model: {clip_model_name}")
        
        self.image_transform = Compose(
            ([SquarePad()] if image_size is not None else [])
            + ([Resize((image_size, image_size))] if image_size is not None else [])
            + [
                ToTensor(),
                Normalize(0.5, 0.5),
            ]
        )
    
    @staticmethod
    def from_image_dataset(
        src_dataset_name: str,
        tokenizer_name: str,
        image_size: int,
        split: str = 'train',
        image_column: str = 'image',
        text_column: str = 'text',
        include_original_image: bool = False,
        limit: int = None,
        **kwargs
    ):
        return ImageLatentsDataset(
            src_dataset_name,
            tokenizer_name=tokenizer_name,
            image_size=image_size,
            split=split,
            image_column=image_column,
            text_column=text_column,
            include_original_image=include_original_image,
            limit=limit,
            **kwargs
        )
    
    def to_hf_dataset(self):
        mapping = {k: [] for k in self[0]} if self else {}
        for item in tqdm(self, desc="Converting to HF Dataset"):
            for k, v in item.items():
                mapping[k].append(v)

        if self.include_original_image:
            # print(f"first row of src_dataset: {self.src_dataset[0]}")
            mapping['image'] = [self.src_dataset[i][self.image_column] for i in range(len(self))]
            # print("Image type:", type(mapping['image'][0]))

        features_dict = {
            "image_emb": Array3D(shape=(4, self.image_size, self.image_size), dtype="float32"),  # TODO: this is hard coded
            "text_emb": Array2D(shape=(1, 768), dtype="float32"),  # TODO: this is hard coded
            "text": Value("string")
        }
        if self.include_original_image:
            features_dict['image'] = Image()
        return HFDataset.from_dict(mapping, features=Features(features_dict))

    def _transform(self, example):
        # Use the VAE and text encoder to get the latents and text encodings
        import clip

        with torch.no_grad():
            img = self.image_transform(example[self.image_column])
            img = img.unsqueeze(0).to(self.device)
            if img.shape[1] == 1:
                try:
                    img = img.repeat(1, 3, 1, 1)
                except:
                    print(f"Error: {img.shape}")
                    raise
            latents = self.vae.encode(img).latent_dist.sample().detach().cpu().numpy().astype(np.float32)[0]

            if self.text_column in example:
                text_encoding = self.clip_model.encode_text(
                    clip.tokenize(example[self.text_column], truncate=True).to(self.device)
                ).detach().cpu().numpy().astype(np.float32)
            else:
                text_encoding = None

        if text_encoding is not None:
            return {
                'image_emb': latents,
                'text_emb': text_encoding,
                'text': example[self.text_column]
            }
        else:
            return {
                'image_emb': latents,
            }

    def __getitem__(self, idx):
        return self._transform(self.src_dataset[idx])
    
    def __len__(self):
        if self.limit is not None:
            return self.limit
        return len(self.src_dataset)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


if __name__ == "__main__":
    # TODO: add the CLIP name, VAE name etc to the huggingface dataset info.
    from datasets import DatasetDict, Dataset as HFDataset
    # ds_name = 'zzsi/afhq64_16k'
    # text_column = 'text'
    # ds = ImageLatentsDataset.from_image_dataset(ds_name, 'madebyollin/sdxl-vae-fp16-fix', 64, 1.0, split='train')
    ds_name = 'eltorio/ROCOv2-radiology'
    text_column = 'caption'
    resolution = 512
    limit = None
    include_original_image = True
    ds = ImageLatentsDataset.from_image_dataset(
        ds_name, 'madebyollin/sdxl-vae-fp16-fix', resolution, split='train', text_column=text_column, include_original_image=include_original_image, limit=limit)
    val_ds = ImageLatentsDataset.from_image_dataset(
        ds_name, 'madebyollin/sdxl-vae-fp16-fix', resolution, split='validation', text_column=text_column, include_original_image=include_original_image, limit=limit)
    print(f"{ds_name}: {len(ds)} examples")
    first_item = ds[0]
    for k, v in first_item.items():
        if hasattr(v, 'shape'):
            print(k, v.shape, v.dtype)
        else:
            print(k, v)
    
    # upload to huggingface
    dataset_dict = DatasetDict({
        'train': ds.to_hf_dataset(),
        'val': val_ds.to_hf_dataset()
    })

    # print(dataset_dict['train'][0]['image'])

    print("Pushing to hub...")
    dataset_dict.push_to_hub(repo_id='zzsi/roco_v2_radiology_latents_sdxl_blip2', max_shard_size="1GB")
    print("Done")