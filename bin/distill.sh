#!/usr/bin/env bash
#
# Consistency Distillation training
# Distills a pre-trained diffusion model into a few-step consistency model.

# if there is an env.sh file, source it
if [ -f bin/env.sh ]; then
	source bin/env.sh
fi

# This is the username in Dockerfile.
USER=nanodiffusion
GPU_DEVICES=${GPU_DEVICES:-0}  # default GPU idx
DOCKER_ARGS=${DOCKER_ARGS:-""}

# Create a data/container_cache directory if it doesn't exist
mkdir -p data/container_cache

docker run --runtime nvidia -it --rm $DOCKER_ARGS \
	--shm-size 16G \
	--gpus "device=${GPU_DEVICES}" \
	-v $(pwd):/workspace \
	-v $(pwd)/data/container_cache:/home/$USER/.cache \
	-e PYTHONPATH=/workspace/src \
	-e WANDB_API_KEY=$WANDB_API_KEY \
	-e WANDB_PROJECT=$WANDB_PROJECT \
	nanodiffusion \
	python -m src.train_distill "$@"

#
### Example usage:
#
# First train a teacher model:
# bin/train.sh --total_steps 10000 --save_every 5000
#
# Then distill it:
# bin/distill.sh --teacher_checkpoint logs/train/<timestamp>/final_model.pth --total_steps 5000
#
# With latent space:
# bin/distill.sh --teacher_checkpoint logs/train/<timestamp>/final_model.pth \
#   -d zzsi/afhq64_16k_latents_sdxl_blip2 --data_shape 4,8,8 --data_is_latent \
#   --net unet_small --total_steps 5000
