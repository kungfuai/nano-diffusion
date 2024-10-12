#!/usr/bin/env bash
#
# Train model

docker run --runtime nvidia -it \
	-v $(pwd):/workspace \
	-v $(pwd)/data/container_cache:/root/.cache \
	-e WANDB_API_KEY=$WANDB_API_KEY \
	-e WANDB_PROJECT=$WANDB_PROJECT \
	nanodiffusion \
	$@
