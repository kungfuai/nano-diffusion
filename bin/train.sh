#!/usr/bin/env bash
#
# Train model

if { ! [[ $* == *--force* ]] && [[ -n $(git status --porcelain) ]]; }; then
  echo -e "\033[33mDIRTY WORKING TREE!\033[0m"
  echo -e "Please commit your changes to your branch. To commit your changes, run \033[32mgit commit\033[0m. to and save your working tree changes in a stack, run \033[36mgit stash\033[0m. If you want to continue working on these changes later on and bring them back into your working tree from the stack, run \033[36mgit stash pop\033[0m"
else
  source bin/setup_environment.sh
  docker compose run --rm app python -m src.training.training_session "$@"
fi
