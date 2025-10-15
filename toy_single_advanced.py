from greens import GreensFunctionCalculator
import numpy as np
import sympy as sp
from utils import sanitize_vector
from ambiguity import AggregatedAmbiguityError
import sys
#sys.tracebacklimit = 0
from typing import Any, List, Optional, Dict, Tuple, Union
ChoiceKey = Tuple[str, str]
CaseAssumptions = Union[
    List[sp.Basic],
    Dict[str, Any],   # expects keys: "predicates": List[sp.Basic], "choices": Dict[Tuple[str,str], Any]
]
import logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.captureWarnings(True) # optional: route warnings -> logging
log = logging.getLogger(__name__)

########################################################################
# Single-channel toy model in symbolic mode with non-constant potential
########################################################################

# Define symbolic parameters
momentum, omega = sp.symbols("k omega", real=True)
eta = sp.symbols("eta", real=True, positive=True)
L = sp.symbols("L", complex=True)
ferro_m = sp.symbols(
    "m_F", real=True, positive=True)
ferro_coeff = sp.symbols(
    "gamma_F", real=True)
nonferro_m = sp.symbols(
    "m_N", real=True, positive=True)
nonferro_coeff = sp.symbols(
    "gamma_N", real=True)

# Define the Hamiltonian for a single-channel toy model
def ferro_hamiltonian(k):
    def ferro_potential(k):
        return ferro_coeff * k*L
    if type(k)==sp.Matrix:
        k = k[0]
    H_ferro = (k**2) / (2 * ferro_m) + ferro_potential(k)
    return H_ferro
def nonferro_hamiltonian(k):
    def nonferro_potential(k):
        return nonferro_coeff * k*L
    H_nonferro = (k**2) / (2 * nonferro_m) + nonferro_potential(k)
    return H_nonferro

# Ferromagnetic side
ferro_greenscalculator = GreensFunctionCalculator(
    hamiltonian=ferro_hamiltonian,
    identity=sp.Matrix([1]),
    symbolic=True,
    energy_level=omega,
    broadening=eta,
    retarded=True,
    dimension=1)

#ferro_G_k = ferro_greenscalculator.compute_kspace_greens_function(momentum) 
#roots = ferro_greenscalculator.compute_roots_greens_inverse(solve_for=0)

z, z_prime = sp.symbols("z z'", real=True)
show_ambiguity_errors = False
try:
    predicates = [sp.Q.positive(ferro_m), sp.Q.positive(eta)]
    choices = {('im_sign_root', 'lambda_0.root_1.sqrt_form'): False, 
               ('im_sign_root', 'lambda_0.root_0.sqrt_form'): True,
               ('im_sign_root', 'lambda_0.root_1.im_sign'): 1,
                ('im_sign_root', 'lambda_0.root_0.im_sign'): 1
               }
    ferro_G_r = ferro_greenscalculator.compute_rspace_greens_symbolic_1d_along_last_dim(z, z_prime, z_diff_sign=1, full_matrix=True, case_assumptions={"predicates": predicates, "choices": choices})
    print(ferro_G_r)
except AggregatedAmbiguityError as e:
    if show_ambiguity_errors:
        raise
    ambiguities = e.items
    decisions = [False, False] # must be one of the options given or "preds" to use the predicate
    assert len(ambiguities) == len(decisions), "Must provide decision for each ambiguity."

    predicates: List[sp.Basic] = []
    choices: Dict[ChoiceKey, Any] = {}

    for i, a in enumerate(ambiguities):
        # 1) Take explicit predicates when provided
        if a.predicate is not None:
            predicates.append(a.predicate)
            assert decisions[i] == "preds", "Must use predicate if provided."
            continue
        # 2) Fill known choice slots
        key: ChoiceKey = (a.where, a.what)
        assert decisions[i] != "preds", "Must choose an option since no predicate is provided."
        assert decisions[i] in a.options, f"Decision {decisions[i]} not in options {a.options}."
        choices[key] = decisions[i]
    
    case = {"predicates": predicates, "choices": choices}
    log.info("Rerunning real-space calculation with additional assumptions.")
    ferro_G_r = ferro_greenscalculator.compute_rspace_greens_symbolic_1d_along_last_dim(z, z_prime, z_diff_sign=1, case_assumptions=case)
