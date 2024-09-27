#!/usr/bin/env bash
#
# Run Jupyter Lab server on http://localhost:8888/lab/

source bin/setup_environment.sh

docker compose run --rm -p 8888:8888 \
  --entrypoint jupyter app lab \
  --ip=0.0.0.0 \
  --port=8888 \
  --notebook-dir=notebooks \
  --allow-root \
  --NotebookApp.token='' \
  --NotebookApp.password=''
