.PHONY: help setup serve run validate render plan apply deploy revisions promote rollback domain cost stop start teardown list mem mysql mysql-down mysql-shell mysql-tail

EXAMPLE ?= chat
ARGS ?=
CONFIG ?= config.yml
WATCH ?=
HASH ?=

help:
	@echo "make setup                                                install python env, runtimes, and models"
	@echo "make serve                                                run the local LLM server (port 8081)"
	@echo "make serve CONFIG=examples/regression/config.yml          serve a predictor instead"
	@echo "make serve CONFIG=implementations/embeddings/config.yml   serve embeddings + chat"
	@echo "make serve CONFIG=... WATCH=--watch                       hot-reload the model on config/file changes"
	@echo "make run EXAMPLE=chat                                     run from implementations/ or examples/"
	@echo "make run EXAMPLE=sdr ARGS=--dry-run                       pass args through to the example"
	@echo "make validate CONFIG=...                                  check a config against Azure's limits, without Azure"
	@echo "make render CONFIG=...                                    print the Dockerfile generated for this config"
	@echo "make plan CONFIG=...                                      diff the config against the live deployment"
	@echo "make apply CONFIG=... HASH=...                            run the plan that HASH was approved for"
	@echo "make deploy CONFIG=...                                    plan and apply in one step, print the URL"
	@echo "make deploy CONFIG=... ARGS=--rebuild                     force an image rebuild"
	@echo "make stop CONFIG=...                                      deactivate the revision, keep the deployment"
	@echo "make start CONFIG=...                                     undo stop, no rebuild"
	@echo "make teardown CONFIG=...                                  delete the deployment"
	@echo "make list                                                 every deployment and its URL"
	@echo "make revisions CONFIG=...                                 revisions, labels and traffic weights"
	@echo "make promote CONFIG=...                                   make the candidate revision live"
	@echo "make rollback CONFIG=...                                  swap the live label back"
	@echo "make domain CONFIG=...                                    custom domain state and DNS records"
	@echo "make cost CONFIG=...                                      estimated monthly cost"
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

validate:
	@uv run simple-local validate -c $(CONFIG) $(ARGS)

render:
	@uv run simple-local render -c $(CONFIG) $(ARGS)

plan:
	@uv run simple-local plan -c $(CONFIG) $(ARGS)

apply deploy:
	@uv run simple-local apply -c $(CONFIG) $(if $(HASH),--plan-hash $(HASH)) $(ARGS)

revisions:
	@uv run simple-local revisions -c $(CONFIG) $(ARGS)

promote:
	@uv run simple-local promote -c $(CONFIG) $(ARGS)

rollback:
	@uv run simple-local rollback -c $(CONFIG) $(ARGS)

domain:
	@uv run simple-local domain -c $(CONFIG) $(ARGS)

cost:
	@uv run simple-local cost -c $(CONFIG) $(ARGS)

stop:
	@uv run simple-local stop -c $(CONFIG) $(ARGS)

start:
	@uv run simple-local start -c $(CONFIG) $(ARGS)

teardown:
	@uv run simple-local teardown -c $(CONFIG) $(ARGS)

list:
	@uv run simple-local list $(ARGS)

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
