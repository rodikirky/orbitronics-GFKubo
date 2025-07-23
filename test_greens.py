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

def test_create_numeric_greens_function():
    # Use concrete numeric values for parameters
    m, gamma, J, Mx = 1.0, 2.0, 0.5, 0.8
    omega, eta = 1.2, 0.01

    system = OrbitronicHamiltonianSystem(
        mass=m,
        orbital_texture_coupling=gamma,
        exchange_interaction_coupling=J,
        magnetisation=[Mx, 0, 0]
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
    assert isinstance(greens_calculator.I, np.ndarray), "Identity should be a sympy Matrix in symbolic mode"
    assert greens_calculator.symbolic is False, "The system should default to numeric mode with symbolic=False"
    assert greens_calculator.omega == 1.2, "omega should remain unchanged after initiation"
    assert greens_calculator.eta == 0.01, "eta should remain unchanged after initiation"
    assert greens_calculator.q == 1, "It should be q==1 for retarded=True"
    assert greens_calculator.verbose is False, "verbose should default to False"


# ────────────────────────────────
# GF construction in k-space 
# ────────────────────────────────

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

def test_numeric_greens_function_shape():
    # Use concrete numeric values for parameters
    m, gamma, J, Mx = 1.0, 2.0, 0.5, 0.8
    omega, eta = 1.2, 0.01

    system = OrbitronicHamiltonianSystem(
        mass=m,
        orbital_texture_coupling=gamma,
        exchange_interaction_coupling=J,
        magnetisation=[Mx, 0, 0],
        symbolic=False
    )
    greens_calculator = GreensFunctionCalculator(
        hamiltonian=system.get_hamiltonian,
        identity=system.identity,
        symbolic=False,
        energy_level=omega,
        infinitestimal=eta,
        retarded=True
    )
    G = greens_calculator.compute_kspace_greens_function(np.array([0.0, 0.0, 0.0]))  # zero momentum
    assert isinstance(G, np.ndarray), "Expected a NumPy ndarray in numeric mode"
    assert G.shape == (3, 3), "Green's function should be 3x3 in this model"
