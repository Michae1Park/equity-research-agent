import os
from pathlib import Path

import yaml

CONFIG_PATH = Path("config.yaml")


def load(path: Path | None = None) -> dict:
    load_dotenv()
    path = path or Path(os.environ.get("ERA_CONFIG") or CONFIG_PATH)
    return yaml.safe_load(path.read_text())


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))
