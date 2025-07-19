.PHONY: test build publish docs

build:
	uv build

publish:
	 uv publish

test:
	uv run python -m pytest

docs:
	uv run mkdocs build