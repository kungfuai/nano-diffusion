#!/usr/bin/env bash
#
# Build Docker service image(s)

source bin/setup_environment.sh

docker compose build "$@"
