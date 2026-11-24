#!/usr/bin/env python3
"""Collect official medium demonstrations in parallel."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tabx_medium_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
