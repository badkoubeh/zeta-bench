# Convenience targets for zeta-bench.
# Run `make help` to list available targets.

.PHONY: help image train eval eval-pid viz shell test lint lock clean

IMAGE   ?= zeta-bench:latest
COMPUTE ?= cpu        # cpu | small_gpu | large_gpu | multi_gpu
AGENT   ?= sac        # sac | ppo
SEED    ?= 42

help:  ## Show this help.
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage: make \033[36m<target>\033[0m\n\n"} \
	      /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@echo ""
	@echo "Variables (override on the command line):"
	@echo "  COMPUTE  ($(COMPUTE))   compute profile"
	@echo "  AGENT    ($(AGENT))      sac | ppo"
	@echo "  SEED     ($(SEED))       integer seed"
	@echo ""

image:  ## Build the Docker image.
	docker build -t $(IMAGE) .

train:  ## Train inside the container.
	docker compose run --rm train experiments/train.py \
	    compute=$(COMPUTE) agent=$(AGENT) seed=$(SEED)

eval:  ## Run the robustness evaluation sweep.
	docker compose run --rm eval experiments/evaluate_robustness.py \
	    compute=$(COMPUTE) seed=$(SEED)

eval-pid:  ## Run the PID baseline end-to-end and store results locally.
	python experiments/evaluate_pid.py seed=$(SEED)

viz:  ## Run PID eval with time-series PNG + MP4 rendering enabled.
	python experiments/evaluate_pid.py seed=$(SEED) eval_pid.render=true

shell:  ## Interactive bash inside the container.
	docker compose run --rm shell

test:  ## Run pytest inside the container.
	docker compose run --rm train -m pytest tests/ -v

install-hooks:  ## Install local pre-commit hooks (run once after cloning).
	pre-commit install

lint:  ## Lint the source tree locally (no container needed).
	ruff check .
	ruff format --check .

lock:  ## Regenerate requirements.lock for Linux x86_64 / Python 3.12.
	VIRTUAL_ENV=.venv uv pip compile pyproject.toml \
	    --extra train --extra dev \
	    --python-platform x86_64-unknown-linux-gnu \
	    --python-version 3.12 \
	    -o requirements.lock

clean:  ## Remove local caches and build artefacts.
	rm -rf .pytest_cache .ruff_cache .mypy_cache \
	       build dist *.egg-info \
	       $(shell find . -type d -name __pycache__ -not -path './.venv/*')
