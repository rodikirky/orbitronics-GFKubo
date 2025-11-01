from greens import GreensFunctionCalculator
from system import OrbitronicHamiltonianSystem
import numpy as np
import sympy as sp
from utils import sanitize_vector, to_jsonable
#from ambiguity import AggregatedAmbiguityError
from pathlib import Path
import json
import pickle
import logging, os
from logging.handlers import RotatingFileHandler
from datetime import datetime
# region logging
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = Path("logs"); LOG_DIR.mkdir(parents=True, exist_ok=True)
log_path = LOG_DIR / f"run_overhaul_{run_id}.log"   # <-- this is the per-run file name

root = logging.getLogger()
#root.setLevel(os.getenv("LOG_LEVEL", "INFO"))
root.setLevel(logging.DEBUG)  # or INFO in production

# clear pre-existing handlers (prevents duplicates in re-runs)
for h in list(root.handlers):
    root.removeHandler(h)

# Console
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
root.addHandler(ch)

# File (rotating)
fh = RotatingFileHandler(log_path, maxBytes=2_000_000, backupCount=5, encoding="utf-8", delay=True)
fh.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
))
root.addHandler(fh)

logging.captureWarnings(True)
logging.getLogger("sympy").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


# Use module loggers everywhere:
log = logging.getLogger(__name__)
log.info("Logging ready.")
# endregion
#######################################
# Orbitronics system in symbolic mode
#######################################
symbolic_mode = True
mass = sp.symbols("m", real=True, positive=True)
orbital_texture_coupling = sp.symbols("gamma", real=True)
#orbital_texture_coupling = 0
exchange_interaction_coupling = sp.symbols("J", real=True)
#exchange_interaction_coupling = 0 # with J=0 determinant is an even polynomial in k
mag1, mag2, mag3 = sp.symbols("M_1 M_2 M_3", real=True)
magnetisation = sanitize_vector([mag1, mag2, mag3], symbolic=symbolic_mode)

system3D = OrbitronicHamiltonianSystem(mass=mass,
                                       orbital_texture_coupling=orbital_texture_coupling,
                                       exchange_interaction_coupling=exchange_interaction_coupling,
                                       magnetisation=magnetisation,
                                       symbolic=symbolic_mode)
def hamiltonian(momentum):
    H_k = system3D.get_hamiltonian(momentum)
    return H_k
calc = GreensFunctionCalculator(
    hamiltonian=hamiltonian,
    retarded=True,
    dimension=3,)

vals = {
    #omega: 0.8, 
    #gamma: 0.3, 
    #J: 1.0,
    mag1: 0.0, mag2: 0.0, #mag3: 3.0,
    mass: 1.0, 
    #eta: 1e-6,      # small positive η for retarded GF
    #k_x: 0.1, k_y: -0.2
}
# Testing all methods individually
G_inv = calc.greens_inverse() # works
G_k = calc.kspace_greens_function(vals=vals)
#with open(Path("results") / "G_k_novals.pkl", "wb") as f:
#    pickle.dump(G_k, f)