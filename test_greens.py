from greens import GreensFunctionCalculator
import numpy as np
import sympy as sp
from system import OrbitronicHamiltonianSystem

# ────────────────────────────────
# Basic Instantiation
# ────────────────────────────────


def test_create_symbolic_greens_function():
    m, gamma, J, Mx = sp.symbols("m gamma J Mx")
    system = OrbitronicHamiltonianSystem(
        mass=m,
        orbital_texture_coupling=gamma,
        exchange_interaction_coupling=J,
        magnetisation=[Mx, 0, 0],
        symbolic=True
    )
    greens_setup = GreensFunctionCalculator(
        hamiltonian=system.get_hamiltonian,
        identity=system.identity,
        symbolic=system.symbolic,
        energy_level=sp.symbols("omega"),
        infinitestimal=sp.symbols("eta"),
        retarded=True
    )
