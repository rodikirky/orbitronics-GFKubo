from greens import GreensFunctionCalculator
import numpy as np
import sympy as sp
from system import OrbitronicHamiltonianSystem
from utils import invert_matrix, hermitian_conjugate
import pytest

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
    assert isinstance(greens_calculator.I, np.ndarray), "Identity should be a numpy array in numeric mode"
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
    assert isinstance(G, sp.Matrix), "The Greens function should be a sympy matrix in symbolic mode"
    assert G.shape == (3, 3), "Expected shape (3,3) for G"

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

def test_symbolic_identity_hamiltonian():
    """
    For known input-output pair comparison, we use the identity function as a Hamiltonian.
    """
    def identity_hamiltonian(k):
        return sp.eye(3)
    omega, eta = sp.symbols("omega eta")
    calculator = GreensFunctionCalculator(
        hamiltonian=identity_hamiltonian,
        identity=sp.eye(3),
        symbolic=True,
        energy_level=omega,
        infinitestimal=eta,
        retarded=True
    )
    G = calculator.compute_kspace_greens_function(sp.Matrix([0, 0, 0]))
    # Expected result is known
    expected = invert_matrix((omega + sp.I * eta) * sp.eye(3) - sp.eye(3), symbolic=True)
    assert G==expected

def test_numeric_identity_hamiltonian():
    """
    For known input-output pair comparison, we use the identity function as a Hamiltonian.
    """
    def identity_hamiltonian(k):
        return np.eye(3)
    omega, eta = 2.0, 0.1
    calculator = GreensFunctionCalculator(
        hamiltonian=identity_hamiltonian,
        identity=np.eye(3),
        symbolic=False,
        energy_level=omega,
        infinitestimal=eta,
        retarded=True
    )
    G = calculator.compute_kspace_greens_function(np.array([0, 0, 0]))
    # Expected result is known:
    expected = np.linalg.inv((omega + 1j * eta) * np.eye(3) - np.eye(3))
    np.testing.assert_allclose(G, expected)

def test_numeric_retarded_vs_advanced():
    omega, eta = 2.0, 0.01

    def simple_H(k): 
        return np.eye(2)
    
    I = np.eye(2)

    retarded_calc = GreensFunctionCalculator(simple_H, I, symbolic=False, energy_level=omega, infinitestimal=eta, retarded=True)
    advanced_calc = GreensFunctionCalculator(simple_H, I, symbolic=False, energy_level=omega, infinitestimal=eta, retarded=False)

    momentum = np.array([0.0, 0.0, 0.0])
    G_ret = retarded_calc.compute_kspace_greens_function(momentum)
    G_adv = advanced_calc.compute_kspace_greens_function(momentum)

    G_ret_dagger = hermitian_conjugate(G_ret, symbolic=False)

    # Check Hermitian conjugate relationship: G_adv ≈ G_ret†
    np.testing.assert_allclose(G_adv, G_ret_dagger, rtol=1e-10, err_msg="Advanced should be Hermitian conjugate of Retarded")

def test_symbolic_retarded_vs_advanced():
    omega, eta = 2.0, 0.01

    def simple_H(k): 
        return sp.eye(2)
    
    I = sp.eye(2)

    retarded_calc = GreensFunctionCalculator(simple_H, I, symbolic=True, energy_level=omega, infinitestimal=eta, retarded=True)
    advanced_calc = GreensFunctionCalculator(simple_H, I, symbolic=True, energy_level=omega, infinitestimal=eta, retarded=False)

    momentum = np.array([0.0, 0.0, 0.0])
    G_ret = retarded_calc.compute_kspace_greens_function(momentum)
    G_adv = advanced_calc.compute_kspace_greens_function(momentum)

    G_ret_dagger = hermitian_conjugate(G_ret, symbolic=True)

    # Check Hermitian conjugate relationship: G_adv ≈ G_ret†
    assert G_adv == G_ret_dagger, "Advanced should be Hermitian conjugate of Retarded"

# ────────────────────────────────
# Error Handling
# ────────────────────────────────


def test_numeric_noninvertible_matrix_raises():
    def singular_hamiltonian(k):
        return np.eye(3)  # ωI - 0 = ωI → always invertible, so instead:
        # Return identity to cancel ω*I, i.e., ωI - I = 0 for ω=1

    omega = 1.0
    eta = 0.0
    identity = np.eye(3)

    calculator = GreensFunctionCalculator(
        hamiltonian=singular_hamiltonian,
        identity=identity,
        symbolic=False,
        energy_level=omega,
        infinitestimal=eta,
        retarded=True
    )

    with pytest.raises(ValueError, match="not invertible"):
        calculator.compute_kspace_greens_function(np.array([0.0, 0.0, 0.0]))

def test_symbolic_noninvertible_matrix_raises():
    omega, eta = sp.symbols("omega eta", real=True)

    # We want: (omega + i*eta)I - H(k) == zero matrix
    # So set H(k) = (omega + i*eta)*I
    def matched_hamiltonian(k):
        return (omega + sp.I * eta) * sp.eye(2)

    identity = sp.eye(2)

    calculator = GreensFunctionCalculator(
        hamiltonian=matched_hamiltonian,
        identity=identity,
        symbolic=True,
        energy_level=omega,
        infinitestimal=eta,
        retarded=True
    )

    with pytest.raises(ValueError, match="not invertible"):
        calculator.compute_kspace_greens_function(sp.Matrix([0, 0, 0]))