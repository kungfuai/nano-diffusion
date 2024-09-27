#!/usr/bin/env bash

export COMPOSE_DOCKER_CLI_BUILD=1
export DOCKER_BUILDKIT=1
export BUILDKIT_INLINE_CACHE=1
export SERVICE=nano-diffusion 


if [ "$(docker info | grep Runtimes | grep -o nvidia)" == "nvidia" ] &&
  command -v nvidia-smi &>/dev/null &&
  [ "$(nvidia-smi -L | grep -c GPU)" -gt 0 ]; then
  export DOCKER_RUNTIME="nvidia"
else
  export DOCKER_RUNTIME="runc"
fi