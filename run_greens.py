"""
Blueprint runner for GreensFunctionCalculator

Etiquette:
- Library code (greens.py, system.py, utils.py) stays side-effect free.
- Runner owns: logging config, output directory, filenames, saving.

Usage examples:
  python run_greens.py --symbolic --out results/greens_3d --stamp
  python run_greens.py --symbolic --out results/tmp --name test_run --log DEBUG
"""
from __future__ import annotations
import argparse, json, logging
from pathlib import Path
from datetime import datetime

import sympy as sp
import numpy as np

# Project imports
from greens import GreensFunctionCalculator
from system import OrbitronicHamiltonianSystem
from utils import sanitize_vector, save_result

# ----------------------------
# CLI
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Green's function and save results.")
    p.add_argument("--symbolic", action="store_true", help="Run in symbolic mode.")
    p.add_argument("--out", type=Path, default=Path("results/greens_runs"),
                   help="Output directory (runner will create it).")
    p.add_argument("--name", type=str, default="run",
                   help="Base name for files inside the output folder.")
    p.add_argument("--stamp", action="store_true",
                   help="Put results in a timestamped subfolder (YYYYmmdd_HHMMSS).")
    p.add_argument("--log", type=str, default="INFO",
                   choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"],
                   help="Logging level.")
    return p.parse_args()

# ----------------------------
# Logging
# ----------------------------
def setup_logging(level: str) -> None: 
    '''
    Configures the logging module to log messages to the console.

    level options: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
    '''
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.captureWarnings(True) # route warnings -> logging

# ----------------------------
# Output directory + manifest
# ----------------------------
def prepare_output_dir(base: Path, stamp: bool, name: str) -> Path:
    if stamp:
        d = datetime.now().strftime("%Y%m%d")
        t = datetime.now().strftime("%H%M%S")
        out = base / d / f"{t}_{name}"
    else:
        out = base / name
    out.mkdir(parents=True, exist_ok=True)
    return out