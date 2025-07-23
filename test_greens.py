from greens import GreensFunctionCalculator
import numpy as np
import sympy as sp
from system import OrbitronicHamiltonianSystem

# ────────────────────────────────
# Basic Instantiation
# ────────────────────────────────


def test_create_symbolic_greens_function():
    m, gamma, J, Mx, omega, eta = sp.symbols("m gamma J Mx omega eta")

    system = OrbitronicHamiltonianSystem(
        mass=m,
        orbital_texture_coupling=gamma,
        exchange_interaction_coupling=J,
        magnetisation=[Mx, 0, 0],
        symbolic=True
    )
    greens_calculator = GreensFunctionCalculator(
        hamiltonian=system.get_hamiltonian,
        identity=system.identity,
        symbolic=system.symbolic,
        energy_level=omega,
        infinitestimal=eta,
        retarded=True
    )
    # Basic assertions to confirm setup
    assert callable(greens_calculator.H), "Hamiltonian should be callable"
    assert isinstance(greens_calculator.I, sp.Matrix), "Identity should be a sympy Matrix in symbolic mode"
    assert greens_calculator.symbolic is True, "In symbolic mode, it should be symbolic is True"
    assert greens_calculator.omega == omega, "omega should remain unchanged after initiation"
    assert greens_calculator.eta == eta, "eta should remain unchanged after initiation"
    assert greens_calculator.q == 1, "It should be q==1 for retarded=True"
    assert greens_calculator.verbose is False, "verbose should default to False"

def test_symbolic_greens_function_shape():
    m, gamma, J, Mx, omega, eta = sp.symbols("m gamma J Mx omega eta")

    system = OrbitronicHamiltonianSystem(
        mass=m,
        orbital_texture_coupling=gamma,
        exchange_interaction_coupling=J,
        magnetisation=[Mx, 0, 0],
        symbolic=True
    )
    greens_calculator = GreensFunctionCalculator(
        hamiltonian=system.get_hamiltonian,
        identity=system.identity,
        symbolic=system.symbolic,
        energy_level=omega,
        infinitestimal=eta,
        retarded=True
    )
    G = greens_calculator.compute_kspace_greens_function(sp.Matrix([0, 0, 0])) # zero momentum test run
    assert isinstance(G, sp.Matrix)
    assert G.shape == (3, 3)
