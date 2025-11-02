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
root.setLevel(logging.INFO)  # or INFO in production

# clear pre-existing handlers (prevents duplicates in re-runs)
for h in list(root.handlers):
    root.removeHandler(h)

# Console
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
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

# TESTING all methods individually:

G_inv = calc.greens_inverse() # works
#print(G_inv)
#adj_G_inv = calc.adjugate_greens_inverse()
#G_k = calc.kspace_greens_function()
#A_00 = adj_G_inv[0,0]
#num_poly_dc, denom_poly_dc = calc.numerator_denominator_poly(A_ij=A_00, i=0, j=0)
det = calc.determinant_poly()
#poles = calc.conditional_poles()
REQUIRED = calc.required_parameters(G_inv)

J = REQUIRED[0]
mag1 = REQUIRED[1]
mag2 = REQUIRED[2]
mag3 = REQUIRED[3]
eta = REQUIRED[4]
gamma = REQUIRED[5]
k_x = REQUIRED[6]
k_y = REQUIRED[7]
mass = REQUIRED[8]
omega = REQUIRED[9]

vals = {
    omega: 1.0, 
    gamma: 0.1, 
    J: 1.0,
    mag1: 0.0, mag2: 0.0, mag3: 3.0,
    mass: 1.0, 
    eta: 1e-6,      # small positive η for retarded GF
    k_x: 0.1, k_y: -0.2
}
assert len(REQUIRED) == len(vals)
#det_poles = calc.poly_poles(det,vals)
z = sp.symbols("z", real=True, positive=True)
z_prime = sp.symbols("z'", real=True, positive=False)
#G_00 = calc.fourier_entry(0, 0, z, z_prime, vals, lambdified=False)
Gz_fullmatrix = calc.fourier_transform(z, z_prime, vals,lambdified=False)
G_r = calc.rspace_greens_function_last_dim(z, z_prime,vals)
assert G_r == Gz_fullmatrix
#with open(Path("results") / "Gz.pkl", "wb") as f:
#    pickle.dump(Gz_fullmatrix, f)