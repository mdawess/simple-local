.PHONY: help setup serve run

EXAMPLE ?= chat
ARGS ?=
CONFIG ?= config.yml

help:
	@echo "make setup                                     install python env, runtimes, and models"
	@echo "make serve                                     run the local LLM server (port 8081)"
	@echo "make serve CONFIG=examples/regression/config.yml   serve a predictor instead"
	@echo "make run EXAMPLE=chat                           run an example (chat | sdr | regression)"
	@echo "make run EXAMPLE=sdr ARGS=--dry-run             pass args through to the example"

setup:
	@./scripts/setup.sh

serve:
	uv run simple-local serve -c $(CONFIG)

run:
	@./scripts/run.sh $(EXAMPLE) $(ARGS)
