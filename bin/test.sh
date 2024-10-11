#!/usr/bin/env bash
#
# Run unit tests

docker run --runtime nvidia -it \
	-v $(pwd):/workspace \
	-v $(pwd)/data/container_cache:/root/.cache \
	nanodiffusion \
	python -m pytest -p no:warnings "${@:-tests/}"
