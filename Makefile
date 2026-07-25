MAIN := main.py

MYPY_FLAGS := --warn-return-any --warn-unused-ignores \
			  --ignore-missing-imports --disallow-untyped-defs \
			  --check-untyped-defs


install:

run:
	uv run python -m $(MAIN)

debug:
	 -m pdb srcs/main.py

clean:

lint:

lint-strict: