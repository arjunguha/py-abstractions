Use `uv` to run code and tests. See `Makefile` for examples.

## Publishing Process

Increment the version number as appropriate in pyproject.toml. Delete old
builds from dist/. Run uv sync to update the lock file. Commit the changes,
which should have just changes to pyproject.taml and uv.lock. Run `uv build`.

Then, tell the user to run `uv publish` manually. Remind them that in the
interactive prompt, you MUST enter __token__ for the username. I know it says
that, but I don't read what's on the screen. The password is in the keychain.