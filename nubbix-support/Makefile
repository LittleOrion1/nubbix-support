PYTHON = . .venv/bin/activate && python

.PHONY: run eval adversarial

run:
	$(PYTHON) src/main.py --query "Necesito ayuda con mi última factura"

eval:
	$(PYTHON) evals/run_evals.py

adversarial:
	$(PYTHON) evals/run_adversarial.py
