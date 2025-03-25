#!/usr/bin/env bash
#
# Train model

# if there is an env.sh file, source it
if [ -f env.sh ]; then
	source env.sh
fi

# This is the username in Dockerfile.
USER=nanodiffusion
WANDB_PROJECT=tmp

# Create a data/container_cache directory if it doesn't exist
mkdir -p data/container_cache
chmod a+rw -R data/container_cache
# Create a checkpoints directory if it doesn't exist
mkdir -p checkpoints
chmod a+rw -R checkpoints

docker run --runtime nvidia -it --rm \
    --shm-size 16G \
	--gpus 'device=0' \
	-v $(pwd):/workspace \
	-v $(pwd)/checkpoints:/workspace/checkpoints \
	-v $(pwd)/data/container_cache:/home/$USER/.cache \
	-e WANDB_API_KEY=$WANDB_API_KEY \
	-e WANDB_PROJECT=$WANDB_PROJECT \
	docker_lib_5 \
	python train.py $@
