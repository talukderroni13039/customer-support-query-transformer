"""Shared test setup. Default tests never touch real APIs or real models."""

import os

# Must run before `src.main` is imported so no real `.env.*` file (and no API key) is loaded.
# Live tests load the key themselves, inside their own fixture.
os.environ["APP_ENV"] = "test"
os.environ.pop("ANTHROPIC_API_KEY", None)
