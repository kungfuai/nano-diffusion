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

The Docker wrapper scripts mount the repo into the container and reuse the host cache directory by default (`$HOME/.cache`). You can override that location with `HOST_CACHE_DIR=/path/to/cache`.

See [docs/roadmap.md](docs/roadmap.md) for the repo direction and planned workflows.
See [docs/diffusion-training-architecture.md](docs/diffusion-training-architecture.md) for the high-level diffusion training structure.
See [docs/training-algorithms.md](docs/training-algorithms.md) for a comparison of the major training algorithms in this repo.
See [docs/global-to-local-supervision.md](docs/global-to-local-supervision.md) for a note on deriving local stepwise supervision from global objectives, including the connection to RL.

## Conditional flow matching

To train a conditional flow matching model, run:

```bash
bin/run.sh python -m src.train_cfm
```

## Other training entrypoints

Rectified flow:

```bash
bin/run.sh python -m src.train_rf
```

MeanFlow:

```bash
bin/run.sh python -m src.train_meanflow
```

SiT:

```bash
bin/run.sh python -m src.train_sit
```

Useful runtime overrides:

- `GPU_DEVICES=0,1` to choose visible GPU indices for the Docker run
- `HOST_CACHE_DIR=/path/to/cache` to override the mounted host cache location
- `DOCKER_ARGS="..."` to pass extra flags to `docker run`

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
- Training algorithms, efficiency and scaling:
  - [x] DDPM
  - [x] Conditional flow matching
  - [x] Text conditioning
  - [x] Multi-GPU training
  - [x] Mixed precision training
  - [ ] Distillation
  - [ ] [muP Transfer](https://github.com/microsoft/mup)
  - [ ] Support [MosaicML Streaming Dataset](https://docs.mosaicml.com/projects/streaming/en/stable/preparing_datasets/basic_dataset_conversion.html)
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
- Fine-tune, Model editting and test time techniques:
  - [ ] LoRA fine-tuning
  - [ ] [Fast inversion for Rectified Flow](https://github.com/HolmesShuan/FireFlow-Fast-Inversion-of-Rectified-Flow-for-Image-Semantic-Editing/tree/4fb7ca0a16d01e113e1a079bfb9da2b6d8c1e7b8)
  - [ ] [TransPixar](https://wileewang.github.io/TransPixar/): extending a video generation model to RGBA.
- Alternative training algorithms:
  - [x] [SiT](https://arxiv.org/abs/2401.08740): scalable interpolant transformers
  - [x] [MeanFlow](https://arxiv.org/abs/2505.13447): one-step generative modeling with mean flows
  - [ ] [VAR]: scale-wise autoregressive model for discrete tokens, e.g. [minVAR](https://github.com/nreHieW/minVAR)
  - [ ] [FlowAR](https://arxiv.org/html/2412.15205v1): scale-wise autoregressive model with flow matching
  - [ ] [Micro-Diffusion](https://github.com/SonyResearch/micro_diffusion)
  - [x] [minRF](https://github.com/cloneofsimo/minRF/tree/4fc10e0cc8ba976152c7936a1af9717209f03e18/advanced): a rectified-flow trainer inspired by minRF
- Experiment tracking (using `wandb`)
  - [x] Validation loss (this is missing in many other implementations, and much more informative than the training loss)
  - [x] Log samples
  - [x] Log FID score
  - [x] Visualize denoising path with dimension reduction
  - [x] Resume training from a wandb run
  - [ ] Visualize trajectories in latent space with dimension reduction
