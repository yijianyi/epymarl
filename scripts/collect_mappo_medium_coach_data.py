#!/usr/bin/env python3
"""Collect formal trajectories for the selected MAPPO Medium coaches."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tabx_coach_collection import main

if __name__ == "__main__":
    raise SystemExit(main())
