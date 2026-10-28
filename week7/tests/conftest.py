"""Make the week7 modules importable and force CPU JAX during pytest runs."""

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

WEEK_DIR = Path(__file__).resolve().parents[1]
if str(WEEK_DIR) not in sys.path:
    sys.path.insert(0, str(WEEK_DIR))
