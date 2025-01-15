# Nano-Diffusion: a minimal working diffusion model training pipeline

## Quickstart

Prerequisites: GPU (16GB VRAM or above), docker and NVIDIA container toolkit.

```bash
bin/build.sh
bin/train.sh
```

If you like to log to Weights & Biases, you can set the `WANDB_API_KEY` and `WANDB_PROJECT` environment variables, and run:

```bash
bin/train.sh --logger wandb
```

There is a `bin/env.sh.example` file that you can copy to `bin/env.sh` and set the `WANDB_API_KEY` and `WANDB_PROJECT` variables.

Use `--help` to see other options.


## Conditional flow matching

To train a conditional flow matching model, run:

```bash
bin/run.sh python -m src.train_cfm
```

## Features

- Datasets
  - [x] AFHQ (low resolution subset), CIFAR-10
  - [x] Flowers, CelebA-HQ
  - [ ] Video frames from DeepMind Lab videos
  - [ ] Openvid with Cosmos tokens
  - [ ] Comics (TBD)
  - [ ] SAB
  - [ ] LAION-Aesthetics
- Model architectures:
  - [x] UNet (small, medium, large)
  - [x] DiT (tiny, small, base, large)
- Training algorithms and scaling:
  - [x] DDPM
  - [x] Conditional flow matching
  - [x] Text conditioning
  - [x] Multi-GPU training
  - [x] Mixed precision training
  - [ ] [muP Transfer](https://github.com/microsoft/mup)
- Tokenziers (VAE)
  - [ ] [Cosmos]
  - [ ] [Reducio-VAE](https://github.com/microsoft/Reducio-VAE)
- Conditioning
  - VLM (for creating captions as training data)
    - [x] [BLIP-2](https://github.com/salesforce/BLIP)
    - [ ] [QWEN-VL](https://github.com/QwenLM/Qwen-VL)
    - [x] [PaliGemma 2]
  - Text encoder (for creating text embeddings as model input)
    - [x] [CLIP](https://github.com/openai/CLIP)
- Test time techniques:
  - [ ] [Fast inversion for Rectified Flow](https://github.com/HolmesShuan/FireFlow-Fast-Inversion-of-Rectified-Flow-for-Image-Semantic-Editing/tree/4fb7ca0a16d01e113e1a079bfb9da2b6d8c1e7b8)
- Alternative training algorithms:
  - [ ] [VAR]: scale-wise autoregressive model for discrete tokens, e.g. [minVAR](https://github.com/nreHieW/minVAR)
  - [ ] [FlowAR](https://arxiv.org/html/2412.15205v1): scale-wise autoregressive model with flow matching
- Experiment tracking (using `wandb`)
  - [x] Validation loss (this is missing in many other implementations, and much more informative than the training loss)
  - [x] Log samples
  - [x] Log FID score
  - [x] Visualize denoising path with dimension reduction
  - [x] Resume training from a wandb run
  - [ ] Visualize trajectories in latent space with dimension reduction
