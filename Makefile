.PHONY: build world smoke test help
build:
	docker build -t vineyard-disease-sim:dev .
world:
	bash scripts/run-container.sh python simulation/generate_world.py
smoke:
	bash scripts/run-container.sh python scripts/sim_smoke.py --timeout 120
test:
	bash scripts/run-container.sh python -m pytest -q
help:
	bash scripts/run-container.sh python -m vineyard_disease --help
