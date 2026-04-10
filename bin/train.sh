#!/usr/bin/env bash
#
# Train the default diffusion entrypoint inside Docker.

# if there is an env.sh file, source it
if [ -f bin/env.sh ]; then
	source bin/env.sh
fi

# This is the username in Dockerfile.
USER=nanodiffusion
GPU_DEVICES=${GPU_DEVICES:-0}  # default GPU idx
DOCKER_ARGS=${DOCKER_ARGS:-""}
IMAGE=${NANODIFFUSION_IMAGE:-nanodiffusion}
HOST_UID=$(id -u)
HOST_GID=$(id -g)
HOST_CACHE_DIR=${HOST_CACHE_DIR:-$HOME/.cache}
CONTAINER_HOME=/tmp

mkdir -p "$HOST_CACHE_DIR"

DOCKER_TTY_ARGS=(-i)
if [ -t 0 ] && [ -t 1 ]; then
	DOCKER_TTY_ARGS=(-it)
fi

docker run --runtime nvidia "${DOCKER_TTY_ARGS[@]}" --rm $DOCKER_ARGS \
	--shm-size 16G \
	--gpus "device=${GPU_DEVICES}" \
	--user "${HOST_UID}:${HOST_GID}" \
	-w /workspace \
	-v "$PWD":/workspace \
	-v "$HOST_CACHE_DIR":"$CONTAINER_HOME/.cache" \
	-e HOME="$CONTAINER_HOME" \
	-e XDG_CACHE_HOME="$CONTAINER_HOME/.cache" \
	-e XDG_CONFIG_HOME="$CONTAINER_HOME/.config" \
	-e MPLCONFIGDIR="$CONTAINER_HOME/.config/matplotlib" \
	-e PYTHONPATH=/workspace/src \
	-e WANDB_API_KEY=$WANDB_API_KEY \
	-e WANDB_PROJECT=$WANDB_PROJECT \
	$IMAGE \
	python -m src.train_diffusion "$@"

#
### For diagnostic run, do one of the following:
#
# bin/train.sh --validate_every 20 --sample_every 30 --num_denoising_steps 2 --fid_every 50 --save_every 60 --total_steps 61 --num_samples_for_fid 100
# or (with latents)
# bin/train.sh -d zzsi/afhq64_16k_latents_sdxl_blip2 --data_shape 4,8,8 --data_is_latent --net unet_small --validate_every 20 --sample_every 30 --num_denoising_steps 2 --fid_every 50 --save_every 60 --total_steps 61 --num_samples_for_fid 100
# or (with text conditioning)
# bin/train.sh -d zzsi/afhq64_16k_latents_sdxl_blip2 --data_shape 3,64,64 --conditional --net unet_small --validate_every 20 --sample_every 30 --num_denoising_steps 2 --fid_every 50 --save_every 60 --total_steps 61 --num_samples_for_fid 100
# or (with latents and text conditioning)
# bin/train.sh --conditional --data_is_latent -d zzsi/afhq64_16k_latents_sdxl_blip2 --data_shape 4,8,8 --net unet_small --validate_every 20 --sample_every 30 --num_denoising_steps 2 --fid_every 50 --save_every 60 --total_steps 61 --num_samples_for_fid 100
# or (for VDM, with latents and text conditioning)
# bin/train.sh --diffusion_algo vdm --fp16 --accelerator --conditional --data_is_latent -d zzsi/afhq64_16k_latents_sdxl_blip2 --data_shape 4,8,8 --net tld_b2 --validate_every 20 --sample_every 30 --num_denoising_steps 2 --fid_every 50 --save_every 60 --total_steps 61 --use_ema --ema_start_step 10000 --ema_beta 0.9999 --num_samples_for_fid 100
