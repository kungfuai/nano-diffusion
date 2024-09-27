#!/usr/bin/env bash
#
# Run unit tests

source bin/setup_environment.sh

docker compose run --rm --entrypoint python app -m pytest -p no:warnings "${@:-tests/}"
