"""Loads config.toml, the single place for tunable values."""

import tomllib
from pathlib import Path

from g1_imitation.robot import REPO_ROOT

DEFAULT_PATH = REPO_ROOT / "config.toml"


def load(path: Path = DEFAULT_PATH) -> dict:
    """The config as nested dicts, one per [section]."""
    with open(path, "rb") as f:
        return tomllib.load(f)
