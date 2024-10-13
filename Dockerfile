FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime


RUN pip install torchvision==0.19.1 numpy==2.1.2 scipy==1.14.1 \
    clean-fid==0.1.35 wandb==0.18.3 timm==1.0.9
# torchdyn for neural ODEs, and POT for optimal transport
RUN pip install torchdyn==1.0.6 POT==0.9.4