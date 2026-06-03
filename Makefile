.PHONY: test typecheck build publish docs

build:
	uv build

publish:
	 uv publish

test:
	uv run python -m pytest

typecheck:
	uv run ty check

docs:
	uv run mkdocs build
