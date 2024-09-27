#!/usr/bin/env bash
#
# Stops and removes docker containers and images

source bin/setup_environment.sh

docker compose down "$@"
