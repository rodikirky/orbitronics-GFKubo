from greens import GreensFunctionCalculator
import numpy as np
import sympy as sp
from utils import sanitize_vector

################################################################
# Single-channel toy model in symbolic mode
################################################################

# Define symbolic parameters
momentum, omega, eta = sp.symbols("k omega eta", real=True)
ferro_m, ferro_potential = sp.symbols(
    "m_F V_F", real=True)
nonferro_m, nonferro_potential = sp.symbols(
    "m_N V_N", real=True)

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
    dimension=1,
    verbose=True)

#ferro_greenscalculator.info() # correct
#ferro_G_k = ferro_greenscalculator.compute_kspace_greens_function(momentum) # correct
#_, eigenvalues, _ = ferro_greenscalculator.compute_eigen_greens_inverse(momentum) # correct
roots = ferro_greenscalculator.compute_roots_greens_inverse(solve_for=0)


# Non-ferromagnetic side
nonferro_greenscalculator = GreensFunctionCalculator(
    hamiltonian=nonferro_hamiltonian,
    identity=sp.Matrix([1]),
    symbolic=True,
    energy_level=omega,
    broadening=eta,
    retarded=True,
    dimension=1,
    verbose=True)

#nonferro_greenscalculator.info() # correct
