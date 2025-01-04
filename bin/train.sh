#!/usr/bin/env bash
#
# Train model

# if there is an env.sh file, source it
if [ -f bin/env.sh ]; then
	source bin/env.sh
fi

# This is the username in Dockerfile.
USER=nanodiffusion
GPU_DEVICES=${GPU_DEVICES:-0}  # default GPU idx

# Create a data/container_cache directory if it doesn't exist
mkdir -p data/container_cache
# You may need to run this command to fix the permissions:
# sudo chmod a+rw -R data/container_cache

docker run --runtime nvidia -it --rm \
	--shm-size 16G \
	--gpus "device=${GPU_DEVICES}" \
	-v $(pwd):/workspace \
	-v $(pwd)/data/container_cache:/home/$USER/.cache \
	-e WANDB_API_KEY=$WANDB_API_KEY \
	-e WANDB_PROJECT=$WANDB_PROJECT \
	nanodiffusion \
	python -m src.train $@

# For diagnostic run, pass in the following args:
# --validate_every 20 --sample_every 30 --num_denoising_steps 2 --fid_every 50 --save_every 60 --total_steps 61
# or
# --net dit_t0 --validate_every 20 --sample_every 30 --num_denoising_steps 2 --fid_every 50 --save_every 60 --total_steps 61
