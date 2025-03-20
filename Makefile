.PHONY: setup test clean

VENV_DIR = .venv
PYTHON = $(VENV_DIR)/bin/python
PIP = $(VENV_DIR)/bin/pip

setup:
	python3 -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

test: setup
	$(PYTHON) -m pytest test_dataset.py

clean:
	rm -rf $(VENV_DIR)
