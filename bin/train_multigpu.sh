#!/usr/bin/env bash
#
# Train model

# if there is an env.sh file, source it
if [ -f bin/env.sh ]; then
	source bin/env.sh
fi

USER=nanodiffusion

docker run --runtime nvidia -it --rm \
	--name nanodiffusion_training \
	--gpus '"device=0,1"' \
	-v $(pwd):/workspace:rw \
	-v $(pwd)/data/container_cache:/$USER/.cache:rw \
	-e WANDB_API_KEY=$WANDB_API_KEY \
	-e WANDB_PROJECT=$WANDB_PROJECT \
	-e OMP_NUM_THREADS=1 \
	-e TORCHELASTIC_ERROR_FILE=logs/error.json \
	nanodiffusion \
	torchrun --standalone \
		--nnodes 1 \
		--nproc-per-node gpu \
		--redirects=3 \
		--log-dir ./logs/multigpu \
		-m src.train_multigpu $@

# To do a quick diagnostic run, pass in the following args:
# bash bin/train_multigpu.sh -d zzsi/afhq64_16k --validate_every 20 --sample_every 30 --num_denoising_steps 2 --fid_every 50