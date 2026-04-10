#!/usr/bin/env bash
#
# Run an arbitrary multi-GPU command inside the training container.

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

docker run --runtime nvidia "${DOCKER_TTY_ARGS[@]}" \
	--shm-size 16G \
	--gpus '"device=1,2"' \
	--user "${HOST_UID}:${HOST_GID}" \
	-w /workspace \
	-v "$PWD":/workspace \
	-v "$HOST_CACHE_DIR":"$CONTAINER_HOME/.cache" \
	-e HOME="$CONTAINER_HOME" \
	-e XDG_CACHE_HOME="$CONTAINER_HOME/.cache" \
	-e XDG_CONFIG_HOME="$CONTAINER_HOME/.config" \
	-e MPLCONFIGDIR="$CONTAINER_HOME/.config/matplotlib" \
	-e WANDB_API_KEY=$WANDB_API_KEY \
	-e WANDB_PROJECT=$WANDB_PROJECT \
	-e OMP_NUM_THREADS=1 \
	-e TORCHELASTIC_ERROR_FILE=logs/error.json \
	$IMAGE \
	$@
