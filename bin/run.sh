#!/usr/bin/env bash
#
# Run an arbitrary command inside the training container.

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
	$IMAGE "$@"

### For diagnostic run of flow matching, do one of the following:
# bin/run.sh python -m src.train_cfm --validate_every 20 --sample_every 30 --num_denoising_steps 2 --fid_every 50 --save_every 60 --total_steps 61
