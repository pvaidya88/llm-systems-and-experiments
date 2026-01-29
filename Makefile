.PHONY: setup bench-smoke bench-squad test

setup:
	pip install -r requirements.txt -r requirements-dev.txt

bench-smoke:
	cd rlm-demo && python -m bench.run --config bench/configs/smoke.json
	cd rlm-demo && python -m bench.compare_runs --baseline bench/baselines/smoke_summary.json --candidate artifacts/smoke/summary.json --thresholds bench/configs/smoke_thresholds.json --run-gates

bench-squad:
	cd rlm-demo && python -m bench.run --config bench/configs/squad_public.json

test:
	pytest rlm-demo/tests
