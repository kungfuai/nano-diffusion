#!/usr/bin/env bash
#
# Run docker container in bash shell session

source bin/setup_environment.sh

docker compose run --rm --entrypoint bash --service-ports "$@"
