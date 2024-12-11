#!/usr/bin/env bash
#
# Train model
source bin/env.sh

docker run --runtime nvidia -it \
    --gpus '"device=1,2"' \
	-v $(pwd):/workspace \
	-v $(pwd)/data/container_cache:/root/.cache \
	-e WANDB_API_KEY=$WANDB_API_KEY \
	-e WANDB_PROJECT=$WANDB_PROJECT \
	-e OMP_NUM_THREADS=1 \
	-e TORCHELASTIC_ERROR_FILE=logs/error.json \
	nanodiffusion \
	$@
