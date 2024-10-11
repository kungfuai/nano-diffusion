#!/usr/bin/env bash
#
# Run docker container in bash shell session

docker run --runtime nvidia -it \
	-v $(pwd):/workspace \
	-v $(pwd)/data/container_cache:/root/.cache \
	-e WANDB_API_KEY=$WANDB_API_KEY \
	-e WANDB_PROJECT=$WANDB_PROJECT \
	nanodiffusion \
	python -m src.train "$@"
