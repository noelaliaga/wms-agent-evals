PYTHON  ?= python3
VENV    ?= .venv
BIN     := $(VENV)/bin
MODELS  ?=
PROMPTS ?= baseline,ask_before_assume
REPEATS ?= 1
TASKS   ?=
OUT     ?= out

empty :=
space := $(empty) $(empty)
comma := ,
MODEL_LIST := $(subst $(space),$(comma),$(strip $(MODELS)))
TASK_ARG   := $(if $(strip $(TASKS)),--tasks $(TASKS),)

.PHONY: install test lint typecheck format validate eval eval-offline check-offline \
        report-offline eval-live check clean

install:  ## .venv with mcp-logistica (git submodule) and this package (PYTHON=python3.12 to pick one)
	@test -f vendor/mcp-logistica/pyproject.toml || (echo "vendor/mcp-logistica is empty: run 'git submodule update --init'" && exit 2)
	@$(PYTHON) -c 'import sys; v = sys.version.split()[0]; sys.exit(0 if sys.version_info >= (3, 11) else f"Python >= 3.11 required, $(PYTHON) is {v}. Try: make install PYTHON=python3.12")'
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/python -m pip install -e vendor/mcp-logistica
	$(BIN)/python -m pip install -e '.[dev]'

test:
	$(BIN)/pytest

lint:  ## ruff lint + format check + mypy strict
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .
	$(BIN)/mypy

typecheck:
	$(BIN)/mypy

format:
	$(BIN)/ruff format .
	$(BIN)/ruff check --fix .

validate:  ## dataset, recordings and prompts load and line up
	$(BIN)/wms-evals validate

eval: eval-offline

eval-offline:  ## mock provider, synthetic recordings, no API calls -> out/offline
	$(BIN)/wms-evals run --provider mock --prompts $(PROMPTS) $(TASK_ARG) --out $(OUT)/offline

check-offline: eval-offline  ## fresh offline run must match the committed results (latency ignored)
	$(BIN)/wms-evals compare reports/offline/results.json $(OUT)/offline/results.json

report-offline:  ## regenerate the committed offline results and report
	$(BIN)/wms-evals run --provider mock --out reports/offline

eval-live:  ## YOUR keys, paid calls: make eval-live MODELS="openai/<m> anthropic/<m> gemini/<m>"
	@test -n "$(MODEL_LIST)" || (echo 'set MODELS, e.g. make eval-live MODELS="openai/<model> anthropic/<model> gemini/<model>"' && exit 2)
	@$(BIN)/python -c 'import litellm' 2>/dev/null || (echo "run: $(BIN)/python -m pip install -e '.[live]'" && exit 2)
	$(BIN)/wms-evals run --provider live --models $(MODEL_LIST) --prompts $(PROMPTS) \
		--repeats $(REPEATS) $(TASK_ARG) --out $(OUT)/live

check: lint test validate check-offline

clean:
	rm -rf .mypy_cache .pytest_cache .ruff_cache build dist src/*.egg-info $(OUT)
