from greens import GreensFunctionCalculator
import numpy as np
import sympy as sp
from utils import sanitize_vector
from ambiguity import build_case_assumptions_from_ledger, AggregatedAmbiguityError
import sys
sys.tracebacklimit = 0
import logging
logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.captureWarnings(True) # optional: route warnings -> logging
log = logging.getLogger(__name__)

################################################################
# Single-channel toy model in symbolic mode
################################################################

# Define symbolic parameters
momentum, omega = sp.symbols("k omega", real=True)
eta = sp.symbols("eta", real=True, positive=True)
ferro_m = sp.symbols(
    "m_F", real=True, positive=True)
ferro_potential = sp.symbols(
    "V_F", real=True)
nonferro_m = sp.symbols(
    "m_N", real=True, positive=True)
nonferro_potential = sp.symbols(
    "V_N", real=True)

# Define the Hamiltonian for a single-channel toy model
def ferro_hamiltonian(k):
    if type(k)==sp.Matrix:
        k = k[0]
    H_ferro = (k**2) / (2 * ferro_m) + ferro_potential
    return H_ferro
def nonferro_hamiltonian(k):
    H_nonferro = (k**2) / (2 * nonferro_m) + nonferro_potential
    return H_nonferro

# ────────────────────────────────
# 1) Translation-invariant systems
# ────────────────────────────────
# Ferromagnetic side
ferro_greenscalculator = GreensFunctionCalculator(
    hamiltonian=ferro_hamiltonian,
    identity=sp.Matrix([1]),
    symbolic=True,
    energy_level=omega,
    broadening=eta,
    retarded=True,
    dimension=1)

#ferro_greenscalculator.info() # correct
#ferro_G_k = ferro_greenscalculator.compute_kspace_greens_function(momentum) # correct
#_, eigenvalues, _ = ferro_greenscalculator.compute_eigen_greens_inverse(momentum) # correct
#roots = ferro_greenscalculator.compute_roots_greens_inverse(solve_for=0) # correct

z, z_prime = sp.symbols("z z'", real=True)
physical_preds = [
        sp.Q.positive(omega - ferro_potential),   # choose propagating regime in F
        sp.Q.positive(ferro_m),                   # already true by symbol, fine to include
        sp.Q.positive(eta)
    ]
ferro_G_r = ferro_greenscalculator.compute_rspace_greens_symbolic_1d_along_last_dim(z, z_prime, z_diff_sign=1, case_assumptions=physical_preds)
'''
try:
    ferro_G_r = ferro_greenscalculator.compute_rspace_greens_symbolic_1d_along_last_dim(z, z_prime, z_diff_sign=1, case_assumptions=physical_preds)
except AggregatedAmbiguityError as e:
    # 1) Propose predicates you *know* are intended for this run:
    physical_preds = [
        sp.Q.positive(omega - ferro_potential),   # choose propagating regime in F
        sp.Q.positive(ferro_m),                   # already true by symbol, fine to include
        sp.Q.positive(eta)
    ]

    # 2) Turn ledger items into case assumptions (accept ConditionSet or not as you like)
    case = build_case_assumptions_from_ledger(
        e.items,
        default_halfplane=+1,           # close in upper half-plane for z>z'
        accept_condition_sets=True,     # or False to force you to add more predicates
        prefer_constant_poly_edgecase=None
    )
    # 3) Merge your own predicates
    case["predicates"] = physical_preds + case["predicates"]

    # 4) Rerun with the resolved assumptions
    log.info("Rerunning real-space calculation with additional assumptions.")
    ferro_G_r = ferro_greenscalculator.compute_rspace_greens_symbolic_1d_along_last_dim(
        z, z_prime, z_diff_sign=1, case_assumptions=case
    )

'''
# Non-ferromagnetic side
#nonferro_greenscalculator = GreensFunctionCalculator(
#    hamiltonian=nonferro_hamiltonian,
#    identity=sp.Matrix([1]),
#    symbolic=True,
#    energy_level=omega,
#    broadening=eta,
#    retarded=True,
#    dimension=1,
#    verbose=True)

#nonferro_greenscalculator.info() # correct
