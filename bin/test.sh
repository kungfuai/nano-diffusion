#!/usr/bin/env bash
#
# Run unit tests

# docker run --runtime nvidia -it \
# 	-v $(pwd):/workspace \
# 	-v $(pwd)/data/container_cache:/root/.cache \
# 	nanodiffusion \
# 	python -m pytest -p no:warnings "${@:-tests/}"

## Run diagnostic training
echo "Running diagnostic training..."
echo "Training DDPM..."
bin/train.sh --validate_every 20 --sample_every 30 --num_denoising_steps 2 --fid_every 50 --save_every 60 --total_steps 61

echo "Training CFM..."
bin/run.sh python -m src.train_cfm --validate_every 20 --sample_every 30 --num_denoising_steps 2 --fid_every 50 --save_every 60 --total_steps 61