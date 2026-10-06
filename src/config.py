"""Central configuration: paths and random seed.

Override locations without editing code via environment variables:
    INFIL_DATA_DIR    folder holding input files (default: <repo>/data)
    INFIL_OUTPUT_DIR  folder for derived files   (default: <repo>/outputs)
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("INFIL_DATA_DIR", ROOT / "data"))
OUTPUT_DIR = Path(os.environ.get("INFIL_OUTPUT_DIR", ROOT / "outputs"))
FIG_DIR = ROOT / "figures"

for _d in (DATA_DIR, OUTPUT_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42  # used for all sampling and model random_state values
