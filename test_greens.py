from greens import GreensFunctionCalculator
import numpy as np
import sympy as sp
from system import OrbitronicHamiltonianSystem
from utils import invert_matrix, hermitian_conjugate
import pytest
from typing import Union

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

    momentum = np.array([0.0, 0.0]) # since H(k) is constant here, k does not actually matter
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

    momentum = [0.0, 0.0] # since H(k) is constant here, k does not actually matter
    G_ret = retarded_calc.compute_kspace_greens_function(momentum)
    G_adv = advanced_calc.compute_kspace_greens_function(momentum)

    G_ret_dagger = hermitian_conjugate(G_ret, symbolic=True)

    # Check Hermitian conjugate relationship: G_adv ≈ G_ret†
    assert G_adv == G_ret_dagger, "Advanced should be Hermitian conjugate of Retarded"

# ────────────────────────────────
# Eigenbasis computation
# ────────────────────────────────

def test_symbolic_eigenvalues_shape_and_form():
    I = sp.eye(2)
    def simple_symbolic_h(kvec):
        return sp.Matrix([
            [kvec[0], 0],
            [0, -kvec[0]]
        ])
    calc = GreensFunctionCalculator(
        hamiltonian=simple_symbolic_h,
        identity=I,
        symbolic=True,
        energy_level=0,
        infinitestimal=0.1,
        verbose=False
    )
    k = calc.k_symbols
    _, eigenvalues, _ = calc.compute_eigen_greens_inverse(k)
    assert isinstance(eigenvalues, (list, sp.Matrix))
    assert all(isinstance(ev, sp.Basic) for ev in eigenvalues)
    assert len(eigenvalues) == 2

# ────────────────────────────────
# Roots computation
# ────────────────────────────────

def test_roots_return_expected_expressions():
    I = sp.eye(2)
    def simple_symbolic_h(kvec):
        return sp.Matrix([
            [kvec[0], 0],
            [0, -kvec[0]]
        ])
    calc = GreensFunctionCalculator(
        hamiltonian=simple_symbolic_h,
        identity=I,
        symbolic=True,
        energy_level=0,
        infinitestimal=0.0,
        verbose=False
    )
    results = calc.compute_roots_greens_inverse(solve_for=0)
    assert isinstance(results, list)
    assert all(len(pair) == 2 for pair in results)
    assert any(sp.S(0) in sol for _, sol in results if isinstance(sol, sp.FiniteSet))

def test_warns_on_non_polynomial_roots():
    def non_polynomial_hamiltonian(kvec):
        kx, ky, kz = kvec
        return sp.Matrix([
            [sp.sin(kx), 0],
            [0, -sp.sin(kx)]
        ])
    calc = GreensFunctionCalculator(
        hamiltonian=non_polynomial_hamiltonian,
        identity=sp.eye(2),
        symbolic=True,
        energy_level=0,
        infinitestimal=0.0,
        verbose=False
    )
    
    with pytest.warns(UserWarning, match="not polynomial"):
        calc.compute_roots_greens_inverse(solve_for=0)

def test_invalid_solve_for_index_raises_value_error():
    calc = GreensFunctionCalculator(
        hamiltonian=lambda k: sp.Matrix([[k[0], 0], [0, -k[0]]]),
        identity=sp.eye(2),
        symbolic=True,
        energy_level=0,
        infinitestimal=0.1
    )

    with pytest.raises(ValueError, match="solve_for out of range"):
        calc.compute_roots_greens_inverse(solve_for=5)


# ────────────────────────────────
# 1D GF construction in real space
# ────────────────────────────────

def test_rspace_green_integrates_known_form():
    z, z_prime = sp.symbols("z z'", real=True)
    eta = sp.symbols("eta", real=True)

    def H(kvec):
        _, _, kz = kvec
        return sp.Matrix([[kz, 0], [0, -kz]])  # Diagonal, easy test

    calc = GreensFunctionCalculator(
        hamiltonian=H,
        identity=sp.eye(2),
        symbolic=True,
        energy_level=0,
        infinitestimal=eta,
        verbose=False
    )

    result = calc.compute_rspace_greens_symbolic_1d(z, z_prime)
    for label, expr in result:
        assert isinstance(expr, sp.Basic)
        assert "Integral" not in str(expr)

def test_warns_when_integral_cannot_be_evaluated():
    z, z_prime = sp.symbols("z z'", real=True)
    eta = sp.symbols("eta", real=True)

    def H(kvec):
        # A non-polynomial (e.g. transcendental) dispersion: sympy can't integrate this
        _, _, kz = kvec
        return sp.Matrix([[sp.sin(kz), 0], [0, -sp.sin(kz)]])

    calc = GreensFunctionCalculator(
        hamiltonian=H,
        identity=sp.eye(2),
        symbolic=True,
        energy_level=0,
        infinitestimal=eta,
        verbose=False
    )

    with pytest.warns(UserWarning, match="unevaluated"):
        result = calc.compute_rspace_greens_symbolic_1d(z, z_prime)
        assert any(expr.atoms(sp.Integral) for _, expr in result)

def test_result_depends_on_difference_not_absolutes():
    eta = sp.symbols("eta", real=True)

    def H(kvec):
            _, _, kz = kvec
            return sp.Matrix([[kz, 0], [0, -kz]])  # Diagonal, easy test

    calc = GreensFunctionCalculator(
        hamiltonian=H,
        identity=sp.eye(2),
        symbolic=True,
        energy_level=0,
        infinitestimal=eta,
        verbose=False
    )
    
    result1 = calc.compute_rspace_greens_symbolic_1d(z=1, z_prime=0)
    result2 = calc.compute_rspace_greens_symbolic_1d(z=2, z_prime=1)    
    for (label1, expr1), (label2, expr2) in zip(result1, result2):
        assert sp.simplify(expr1 - expr2) == 0

def test_multiple_bands_return_distinct_results():
    z, z_prime = sp.symbols("z z'", real=True)

    def H(kvec):
        kx, ky, kz = kvec
        return sp.Matrix([[kz, 0], [0, 2 * kz]])

    calc = GreensFunctionCalculator(
        hamiltonian=H,
        identity=sp.eye(2),
        symbolic=True,
        energy_level=0,
        infinitestimal=0.1,
        verbose=False
    )

    results = calc.compute_rspace_greens_symbolic_1d(z, z_prime)
    assert results[0][1] != results[1][1]

def test_retarded_greens_function_vanishes_without_poles_in_upper_half_plane():
    z, z_prime = sp.symbols("z z'", real=True)

    def H(kvec):
        _, _, kz = kvec
        return sp.Matrix([[kz]])

    calc = GreensFunctionCalculator(
        hamiltonian=H,
        identity=sp.eye(1),
        symbolic=True,
        energy_level=0,
        infinitestimal=1e-20,  # small eta
        verbose=False
    )

    result = calc.compute_rspace_greens_symbolic_1d(z, z_prime)
    _, G_expr = result[0]

    # Substitute values for z, z′ such that z < z′
    val = G_expr.subs({z: 0, z_prime: 1}).evalf()
    assert abs(val) < 1e-6, f"Expected G(z=0, z′=1) ≈ 0, got {val}"

# ────────────────────────────────
# Verbose output
# ────────────────────────────────

def test_numeric_verbose_output(capsys):
    def dummy_H(k): return np.eye(2)

    calculator = GreensFunctionCalculator(
        hamiltonian=dummy_H,
        identity=np.eye(2),
        symbolic=False,
        energy_level=1.0,
        infinitestimal=0.1,
        verbose=True
    )
    
    momentum = [0.0, 0.0] # since H(k) is constant here, k does not actually matter
    calculator.compute_kspace_greens_function(momentum)

    captured = capsys.readouterr()
    assert "Inversion target" in captured.out

def test_symbolic_verbose_output(capsys):
    def dummy_H(k): return sp.eye(2)
    omega, eta = sp.symbols("omega eta", real=True)
    z, z_prime = sp.symbols("z z'", real=True)

    calculator = GreensFunctionCalculator(
        hamiltonian=dummy_H,
        identity=sp.eye(2),
        symbolic=True,
        energy_level=omega,
        infinitestimal=eta,
        verbose=True
    )
    
    momentum = [0.0, 0.0] # since H(k) is constant here, k does not actually matter
    calculator.compute_kspace_greens_function(momentum)
    calculator.compute_roots_greens_inverse(solve_for=0)
    calculator.compute_rspace_greens_symbolic_1d(z, z_prime)

    captured = capsys.readouterr()
    assert "( ω ± iη - H(k) )" in captured.out
    assert "eigenvalues" in captured.out
    assert "Fourier transform" in captured.out

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

def test_warns_in_numeric_mode_returns_empty_list():
    calc = GreensFunctionCalculator(
        hamiltonian=lambda k: np.array([[k[0], 0], [0, -k[0]]]),
        identity=np.eye(2),
        symbolic=False,
        energy_level=0,
        infinitestimal=0.1,
        verbose=False
    )

    with pytest.warns(UserWarning, match="only supported in symbolic mode"):
        roots = calc.compute_roots_greens_inverse(solve_for=0)
        assert roots == []
    
    with pytest.warns(UserWarning, match="only supported in symbolic mode"):
        result = calc.compute_rspace_greens_symbolic_1d(sp.symbols("z"), sp.symbols("z'"))
        assert result == []