#!/usr/bin/env bash
#
# Start docker services

source bin/setup_environment.sh

docker compose up "$@"
