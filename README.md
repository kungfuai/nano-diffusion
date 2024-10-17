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

Use `--help` to see other options.

## Conditional flow matching

To train a conditional flow matching model, run:

```bash
bin/run.sh python -m src.train_cfm
```

## Features

- Datasets
  - [x] CIFAR-10
  - [ ] Flowers, CelebA-HQ
  - [ ] Video frames from DeepMind Lab videos
  - [ ] Comics (TBD)
  - [ ] SAB
  - [ ] LAION-Aesthetics
- Model architectures:
  - [x] UNet (small, medium, large)
  - [x] DiT (tiny, small, base, large)
- Training algorithms and scaling:
  - [x] DDPM
  - [x] Conditional flow matching
  - [ ] Text conditioning
  - [ ] Multi-GPU training
  - [ ] Mixed precision training
- Experiment tracking (using `wandb`)
  - [x] Validation loss (this is missing in many other implementations, and much more informative than the training loss)
  - [x] Log samples
  - [x] Log FID score
  - [ ] Log Denoising results
  - [ ] Visualize denoising path with dimension reduction
  - [x] Resume training from a wandb run
