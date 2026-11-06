"""Reuse existing FLUX.2 module boundaries with the MPS wall-time recorder."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "flux2"))
from profile_3090_modules import module_instrumentation
