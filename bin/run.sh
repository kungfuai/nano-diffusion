#!/usr/bin/env bash
#
# Train model
source bin/env.sh

docker run --runtime nvidia -it \
    --gpus 'device=0' \
	-v $(pwd):/workspace \
	-v $(pwd)/data/container_cache:/root/.cache \
	-e WANDB_API_KEY=$WANDB_API_KEY \
	-e WANDB_PROJECT=$WANDB_PROJECT \
	nanodiffusion \
	$@
