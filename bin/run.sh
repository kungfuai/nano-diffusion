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
DOCKER_ARGS=${DOCKER_ARGS:-""}

mkdir -p data/container_cache

docker run --runtime nvidia -it --rm $DOCKER_ARGS \
	--shm-size 16G \
    --gpus "device=${GPU_DEVICES}" \
	-v $(pwd):/workspace \
	-v $(pwd)/data/container_cache:/home/$USER/.cache \
	-e PYTHONPATH=/workspace/src \
	-e WANDB_API_KEY=$WANDB_API_KEY \
	-e WANDB_PROJECT=$WANDB_PROJECT \
	nanodiffusion "$@"

### For diagnostic run of flow matching, do one of the following:
# bin/run.sh python -m src.train_cfm --validate_every 20 --sample_every 30 --num_denoising_steps 2 --fid_every 50 --save_every 60 --total_steps 61
