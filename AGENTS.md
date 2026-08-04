Use `uv` to run code and tests. See `Makefile` for examples.

## Publishing Process

Increment the version number as appropriate in pyproject.toml. Delete old
builds from dist/. Run uv sync to update the lock file. Commit the changes,
which should have just changes to pyproject.taml and uv.lock. Run `uv build`.

Then, tell the user to run `uv publish` manually. Remind them that in the
interactive prompt, you MUST enter __token__ for the username. I know it says
that, but I don't read what's on the screen. The password is in the keychain.

## Cursor Cloud specific instructions

This is a pure Python library (`abstractions`); there is no long-running server
or web app to start. "Running" it means exercising the library via `uv run`.

- `uv` is installed under `~/.local/bin` (added to `PATH` in `~/.bashrc`). If a
  shell can't find it, run `export PATH="$HOME/.local/bin:$PATH"`.
- Standard commands live in the `Makefile`: `make test` (pytest), `make typecheck`
  (`ty`), `make docs` (mkdocs build). All are wrappers around `uv run ...`.
- Tests use `spawn`-based `multiprocessing` subprocess actors (see
  `src/abstractions/actor.py`), so they need a working `fork`/`spawn` sandbox and
  bind to `127.0.0.1` ephemeral ports.