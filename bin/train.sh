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

# Create a data/container_cache directory if it doesn't exist
mkdir -p data/container_cache
# You may need to run this command to fix the permissions:
# sudo chmod a+rw -R data/container_cache

docker run --runtime nvidia -it --rm $DOCKER_ARGS \
	--shm-size 16G \
	--gpus "device=${GPU_DEVICES}" \
	-v $(pwd):/workspace \
	-v $(pwd)/data/container_cache:/home/$USER/.cache \
	-e PYTHONPATH=/workspace/src \
	-e WANDB_API_KEY=$WANDB_API_KEY \
	-e WANDB_PROJECT=$WANDB_PROJECT \
	nanodiffusion \
	python -m src.train_diffusion "$@"

#
### For diagnostic run, do one of the following:
#
# bin/train.sh --validate_every 20 --sample_every 30 --num_denoising_steps 2 --fid_every 50 --save_every 60 --total_steps 61
# or (with latents)
# bin/train.sh -d zzsi/afhq64_16k_latents_sdxl_blip2 --data_shape 4,8,8 --data_is_latent --net unet_small --validate_every 20 --sample_every 30 --num_denoising_steps 2 --fid_every 50 --save_every 60 --total_steps 61
# or (with text conditioning)
# bin/train.sh -d zzsi/afhq64_16k_latents_sdxl_blip2 --data_shape 3,64,64 --conditional --net unet_small --validate_every 20 --sample_every 30 --num_denoising_steps 2 --fid_every 50 --save_every 60 --total_steps 61
# or (with latents and text conditioning)
# bin/train.sh --conditional --data_is_latent -d zzsi/afhq64_16k_latents_sdxl_blip2 --data_shape 4,8,8 --net unet_small --validate_every 20 --sample_every 30 --num_denoising_steps 2 --fid_every 50 --save_every 60 --total_steps 61
# or (for VDM, with latents and text conditioning)
# bin/train.sh --diffusion_algo vdm --fp16 --accelerator --conditional --data_is_latent -d zzsi/afhq64_16k_latents_sdxl_blip2 --data_shape 4,8,8 --net tld_b2 --validate_every 20 --sample_every 30 --num_denoising_steps 2 --fid_every 50 --save_every 60 --total_steps 61 --use_ema --ema_start_step 10000 --ema_beta 0.9999


