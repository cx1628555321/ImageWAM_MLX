"""Compatibility entry; implementation moved to scripts/m5pro."""
import runpy
import sys
from pathlib import Path
entry = Path(__file__).resolve().parents[1] / "m5pro" / "validate_io.py"
sys.path.insert(0, str(entry.parent))
if __name__ == "__main__":
    runpy.run_path(str(entry), run_name="__main__")
