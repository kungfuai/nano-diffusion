FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# Non-root user
RUN useradd -m -s /bin/bash -G sudo -u 1000 nanodiffusion
USER nanodiffusion

RUN pip install torchvision==0.19.1 numpy==2.1.2 scipy==1.14.1 \
    clean-fid==0.1.35 wandb==0.18.3 timm==1.0.9 datasets==3.0.2
# torchdyn for neural ODEs, and POT for optimal transport
RUN pip install torchdyn==1.0.6 POT==0.9.4
# RUN pip install diffusers  # optional

