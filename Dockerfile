FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

RUN apt-get update -y && apt-get install -y git build-essential && apt-get clean && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN useradd -m -s /bin/bash -G sudo -u 1000 nanodiffusion
USER nanodiffusion

RUN pip install torchvision numpy scipy \
    clean-fid wandb timm datasets
# torchdyn for neural ODEs, and POT for optimal transport
RUN pip install torchdyn POT
RUN pip install transformers einops
RUN pip install diffusers  # for loading pretrained models
RUN pip install git+https://github.com/openai/CLIP.git  # for loading CLIP models

RUN pip install accelerate
