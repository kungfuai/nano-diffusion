# Nano-Diffusion Roadmap

## Goal

Make this repo useful for two audiences:

- ML practitioners and startups who want ready-to-use diffusion workflows
- learners who want to understand diffusion and practical training recipes

The repo should let a user do most common diffusion work without rebuilding the stack from scratch.

## Product Direction

The repo should support two primary jobs:

1. Use diffusion productively
2. Learn diffusion clearly

That means we should optimize for:

- reproducible Docker-first workflows
- a small number of blessed, ready-to-use training paths
- clear progression from simple pretraining to practical fine-tuning and deployment-oriented inference
- code structure that teaches, rather than fragments

## Workflow Model

Pretraining, fine-tuning, and distillation should not be forced into one flat command.

Recommended public command surface:

```bash
python -m src.train pretrain ...
python -m src.train finetune ...
python -m src.train distill ...

python -m src.sample ...
python -m src.eval ...
python -m src.data ...
```

Rationale:

- `pretrain` covers the current family of small-scale training algorithms
- `finetune` has different semantics: initialization, trainable parameters, adapter choices, and data assumptions
- `distill` is a separate workflow with teacher/student logic and different success criteria
- `sample`, `eval`, and `data` should be first-class workflows rather than hidden utilities

Within `pretrain`, algorithm variants should be unified behind a shared interface:

```bash
python -m src.train pretrain --method ddpm
python -m src.train pretrain --method cfm
python -m src.train pretrain --method rf
python -m src.train pretrain --method meanflow
python -m src.train pretrain --method sit
```

## What “Done” Looks Like

A strong version of this repo should let a user:

- train a small diffusion model from scratch
- train a flow-matching or rectified-flow model from scratch
- fine-tune a model on custom data
- run LoRA-based adaptation
- evaluate quality and speed
- distill a teacher into a faster sampler
- generate samples from checkpoints in a deployable workflow
- follow a clear educational path from DDPM to practical latent diffusion

## Priorities

## Now

1. Unify current pretraining entrypoints under `src.train pretrain`
2. Reduce duplicated training/runtime/config code
3. Keep Docker workflows reliable and reproducible
4. Maintain a small set of diagnostic training runs that always work

## Next

1. Add LoRA fine-tuning
2. Add a clean latent-diffusion happy path
3. Add standard evaluation reports and recipe presets
4. Add practical sampling workflows from checkpoints

## Later

1. Add distillation workflows for fast inference
2. Add inpainting and image-to-image workflows
3. Add serving and deployment-oriented recipes
4. Add a structured learning path with docs that map directly to code

## Feature Additions We Want

### Training

- unified pretraining entrypoint
- LoRA fine-tuning
- full fine-tuning where appropriate
- distillation
- stronger resume/export/checkpoint semantics

### Data

- local folder to training dataset pipeline
- captioning / metadata preparation workflow
- latent precompute workflow
- dataset validation tooling

### Sampling and Inference

- sample from a checkpoint
- batch generation
- inpainting
- image-to-image
- export path for deployment

### Evaluation

- FID and related image quality metrics
- speed / latency benchmarking
- side-by-side run comparison
- reproducible report generation

### Education

- DDPM from scratch
- flow matching and rectified flow
- latent diffusion in practice
- fine-tuning recipes
- distillation recipes
- practical troubleshooting guides

## Rules For README Checkboxes

To mark a feature as implemented in the README:

- it should exist in code
- it should be reasonably usable from the main Docker workflow
- it should be something we would be comfortable recommending

The README should stay short and user-facing. This file should carry the longer-term roadmap and architectural direction.
