from system import OrbitronicHamiltonianSystem
import numpy as np
import sympy as sp
import pytest

# ────────────────────────────────
# Basic Instantiation 
# ────────────────────────────────

def test_create_numeric_system():
    system = OrbitronicHamiltonianSystem(
        mass=1.0,
        orbital_texture_coupling=1.0,
        exchange_interaction_coupling=1.0,
        magnetisation=[1, 0, 0],
        symbolic=False
    )
    assert isinstance(system.mass, float), "Expected 'mass' to be a float in numeric mode."
    assert isinstance(system.gamma, float), "Expected 'gamma' to be a float in numeric mode."
    assert isinstance(system.J, float), "Expected 'J' to be a float in numeric mode."

    assert isinstance(system.M, np.ndarray), "Expected 'magnetisation' to be an np.ndarray in numeric mode."
    assert system.M.shape == (3,), "Expected 'mass' be an np.array of (3,) shape in numeric mode."
    assert np.allclose(system.M, [1, 0, 0]), "Expected M to stay equal to magnetisation input."

    assert isinstance(system.basis, np.ndarray), "Expected 'basis' to be a np.ndarray in numeric mode."
    assert system.basis.shape == (3, 3), "Expected basis shape (3,3)."
    assert np.allclose(system.basis, np.eye(3)), "Expected default basis as identity."

    assert system.symbolic is False, "Expected numeric mode."

def test_create_symbolic_system():
    m, gamma, J, Mx = sp.symbols("m gamma J Mx")
    system = OrbitronicHamiltonianSystem(
        mass=m,
        orbital_texture_coupling=gamma,
        exchange_interaction_coupling=J,
        magnetisation=[Mx, 0, 0],
        symbolic=True
    )
    assert isinstance(system.mass, sp.Basic), "Expected 'mass' to be a sympy symbol in symbolic mode."
    assert isinstance(system.gamma, sp.Basic), "Expected 'gamma' to be a sympy symbol in symbolic mode."
    assert isinstance(system.J, sp.Basic), "Expected 'J' to be a sympy symbol in symbolic mode."

    assert isinstance(system.M, sp.Matrix), "Expected 'M' to be a sympy Matrix in symbolic mode."
    assert system.M.shape == (3, 1), "Expected magnetisation shape (3,1)."
    assert system.M[0] == Mx, "Expected M to stay equal to magnetisation input. M[0] changed."
    assert system.M[1] == 0, "Expected M to stay equal to magnetisation input. M[1] changed."
    assert system.M[2] == 0, "Expected M to stay equal to magnetisation input. M[2] changed."

    assert isinstance(system.basis, sp.Matrix)
    assert system.basis.shape == (3, 3), "Expected basis shape (3,3)."
    assert system.basis == sp.eye(3), "Expected default basis as identity."

    assert system.symbolic is True, "Expected symbolic mode."

# ────────────────────────────────
# Hamiltonian Construction
# ────────────────────────────────

def test_numeric_hamiltonian_shape():
    system = OrbitronicHamiltonianSystem(1.0, 1.0, 0.0, [0, 0, 0], symbolic=False)
    H = system.get_hamiltonian([1.0, 0.0, 0.0])
    assert H.shape == (3, 3), "Expected Hamiltonian shape (3,3)."
    assert isinstance(H, np.ndarray), "Expected Hamiltonian to be a numpy array in numeric mode."

def test_symbolic_hamiltonian_entries():
    m, gamma, J, Mx = sp.symbols("m gamma J Mx")
    system = OrbitronicHamiltonianSystem(
        mass=m,
        orbital_texture_coupling=gamma,
        exchange_interaction_coupling=J,
        magnetisation=[Mx, 0, 0],
        symbolic=True
    )
    kx = sp.symbols("kx")
    H = system.get_hamiltonian([kx, 0, 0])
    assert H.shape == (3, 3), "Expected Hamiltonian shape (3,3)."
    for i, entry in enumerate(H): # flattens the matrix to a list for checking
        assert isinstance(entry, sp.Basic), f"H[{i}] is invalid: {repr(entry)} (type: {type(entry)})" # checks all entries of H for symbolic consistency
    H_symb = system.get_hamiltonian([kx, 0, 0])
    assert H_symb.shape == (3, 3), "Expected symbolic Hamiltonian shape (3,3)."
    for i, entry in enumerate(H_symb): # flattens the matrix to a list for checking
        assert isinstance(entry, sp.Basic), f"H_symb[{i}] is invalid: {repr(entry)} (type: {type(entry)})" # checks all entries of H for symbolic consistency
    assert H == H_symb, "Symbolic Hamiltonian should be equal to itself."

# ────────────────────────────────
# Error Handling
# ────────────────────────────────

def test_symbolic_mass_to_numeric_raises():
    m = sp.Symbol("m")
    with pytest.raises(TypeError):
        OrbitronicHamiltonianSystem(
            mass=m, orbital_texture_coupling=1.0,
            exchange_interaction_coupling=1.0, magnetisation=[1, 0, 0],
            symbolic=False
        )

def test_symbolic_magnetisation_in_numeric_mode_raises():
    Mx = sp.Symbol("Mx")
    with pytest.raises(TypeError):
        OrbitronicHamiltonianSystem(
            mass=1.0, orbital_texture_coupling=1.0,
            exchange_interaction_coupling=1.0, magnetisation=[Mx, 0, 0],
            symbolic=False
        )
