#!/usr/bin/env bash
#
# Train the multi-GPU diffusion entrypoint inside Docker.

# if there is an env.sh file, source it
if [ -f bin/env.sh ]; then
	source bin/env.sh
fi

USER=nanodiffusion
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

docker run --runtime nvidia "${DOCKER_TTY_ARGS[@]}" --rm \
	--shm-size 16G \
	--name nanodiffusion_training \
	--gpus '"device=0,1"' \
	--user "${HOST_UID}:${HOST_GID}" \
	-w /workspace \
	-v "$PWD":/workspace:rw \
	-v "$HOST_CACHE_DIR":"$CONTAINER_HOME/.cache":rw \
	-e HOME="$CONTAINER_HOME" \
	-e XDG_CACHE_HOME="$CONTAINER_HOME/.cache" \
	-e XDG_CONFIG_HOME="$CONTAINER_HOME/.config" \
	-e MPLCONFIGDIR="$CONTAINER_HOME/.config/matplotlib" \
	-e WANDB_API_KEY=$WANDB_API_KEY \
	-e WANDB_PROJECT=$WANDB_PROJECT \
	-e OMP_NUM_THREADS=1 \
	-e TORCHELASTIC_ERROR_FILE=logs/error.json \
	$IMAGE \
	torchrun --standalone \
		--nnodes 1 \
		--nproc-per-node gpu \
		--redirects=3 \
		--log-dir ./logs/multigpu \
		-m src.train_multigpu $@

# To do a quick diagnostic run, pass in the following args:
# bash bin/train_multigpu.sh -d zzsi/afhq64_16k --validate_every 20 --sample_every 30 --num_denoising_steps 2 --fid_every 50
