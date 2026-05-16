from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class AppConfig:
    raw: dict[str, Any]

    def get(self, key: str, default: Any = None) -> Any:
        parts = key.split(".")
        value: Any = self.raw
        for part in parts:
            if not isinstance(value, dict) or part not in value:
                return default
            value = value[part]
        return value


def load_config(path: str | Path) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    return AppConfig(raw=payload)
