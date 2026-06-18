PYTHON	= uv run python
SRC		= src

.PHONY: install run debug clean lint lint-strict

install:
	uv sync

run:
	$(PYTHON) -m $(SRC)

debug:
	$(PYTHON) -m pdb -m $(SRC)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

lint:
	flake8 .
	mypy . --warn-return-any --warn-unused-ignores \
	--ignore-missing-imports --disallow-untyped-defs \
	--check-untyped-defs

lint-strict:
	flake8 .
	mypy . --strict
