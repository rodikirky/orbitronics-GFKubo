from __future__ import annotations
import logging
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo  # Python 3.9+
from typing import Callable, Tuple, Any
import importlib

# ----------------------------
# Logging
# ----------------------------
def setup_logging(level: str, out_dir: Path, run_name: str) -> Path:
    """
    Configure logging to write into out_dir with a timestamped filename, e.g. run_20251018-2312.log.
    level options: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
    Returns the full log file path.
    """
    # Europe/Berlin timestamp, safe for filenames
    ts = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%Y%m%d-%H%M%S")
    log_path = out_dir / f"{run_name}_{ts}.log"

    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        filename=log_path,         # <-- unique per run
        encoding="utf-8",
        filemode="a",
    )
    logging.captureWarnings(True) # route warnings -> logging

    return log_path

# ----------------------------
# Dynamic builder loader    
# ----------------------------
def load_builder(builder_spec: str) -> Callable[..., Tuple[Callable[[Any], Any], Any, int]]:
    mod_name, func_name = builder_spec.split(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, func_name)
    if not callable(fn):
        raise TypeError(f"Builder {builder_spec!r} is not callable.")
    return fn