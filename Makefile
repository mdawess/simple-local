.PHONY: help setup serve run mem mysql mysql-down mysql-shell mysql-tail

EXAMPLE ?= chat
ARGS ?=
CONFIG ?= config.yml
WATCH ?=

help:
	@echo "make setup                                                install python env, runtimes, and models"
	@echo "make serve                                                run the local LLM server (port 8081)"
	@echo "make serve CONFIG=examples/regression/config.yml          serve a predictor instead"
	@echo "make serve CONFIG=implementations/embeddings/config.yml   serve embeddings + chat"
	@echo "make serve CONFIG=... WATCH=--watch                       hot-reload the model on config/file changes"
	@echo "make run EXAMPLE=chat                                     run from implementations/ or examples/"
	@echo "make run EXAMPLE=sdr ARGS=--dry-run                       pass args through to the example"
	@echo "make mem                                                  ram/swap and per-model memory usage"
	@echo "make mysql-tail CONFIG=...                                last 20 logged requests (needs logging.mysql)"
	@echo "make mysql-shell CONFIG=...                               sql prompt on the request log"
	@echo "make mysql-down CONFIG=...                                remove the mysql container"

setup:
	@./scripts/setup.sh

serve: mysql
	@if [ -f .env ]; then set -a; . ./.env; set +a; fi; uv run simple-local serve -c $(CONFIG) $(WATCH)

run:
	@./scripts/run.sh $(EXAMPLE) $(ARGS)

mem:
	@uv run python scripts/mem.py

# Starts only when CONFIG has a logging.mysql block pointing at a local host.
mysql:
	@./scripts/mysql.sh $(CONFIG) up

mysql-down:
	@./scripts/mysql.sh $(CONFIG) down

mysql-shell:
	@./scripts/mysql.sh $(CONFIG) shell

mysql-tail:
	@./scripts/mysql.sh $(CONFIG) tail
