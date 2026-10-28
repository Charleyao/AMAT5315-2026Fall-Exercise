import sys
from pathlib import Path

# Allow `pytest` from week8/ to import the local `src` package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
