import os
from pathlib import Path

os.environ["ERA_COMPANIES"] = str(Path(__file__).resolve().parents[1] / "companies-v3.yaml")
