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
    assert isinstance(
        system.mass, float), "Expected 'mass' to be a float in numeric mode."
    assert isinstance(
        system.gamma, float), "Expected 'gamma' to be a float in numeric mode."
    assert isinstance(
        system.J, float), "Expected 'J' to be a float in numeric mode."

    assert isinstance(
        system.M, np.ndarray), "Expected 'magnetisation' to be an np.ndarray in numeric mode."
    assert system.M.shape == (
        3,), "Expected 'mass' be an np.array of (3,) shape in numeric mode."
    assert np.allclose(
        system.M, [1, 0, 0]), "Expected M to stay equal to magnetisation input."

    assert isinstance(
        system.basis, np.ndarray), "Expected 'basis' to be a np.ndarray in numeric mode."
    assert system.basis.shape == (3, 3), "Expected basis shape (3,3)."
    assert np.allclose(system.basis, np.eye(
        3)), "Expected default basis as identity."

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
    assert isinstance(
        system.mass, sp.Basic), "Expected 'mass' to be a sympy symbol in symbolic mode."
    assert isinstance(
        system.gamma, sp.Basic), "Expected 'gamma' to be a sympy symbol in symbolic mode."
    assert isinstance(
        system.J, sp.Basic), "Expected 'J' to be a sympy symbol in symbolic mode."

    assert isinstance(
        system.M, sp.Matrix), "Expected 'M' to be a sympy Matrix in symbolic mode."
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
    system = OrbitronicHamiltonianSystem(
        1.0, 1.0, 0.0, [0, 0, 0], symbolic=False)
    H = system.get_hamiltonian([1.0, 0.0, 0.0])
    assert H.shape == (3, 3), "Expected Hamiltonian shape (3,3)."
    assert isinstance(
        H, np.ndarray), "Expected Hamiltonian to be a numpy array in numeric mode."


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
    for i, entry in enumerate(H):  # flattens the matrix to a list for checking
        # checks all entries of H for symbolic consistency
        assert isinstance(
            entry, sp.Basic), f"H[{i}] is invalid: {repr(entry)} (type: {type(entry)})"
    H_symb = system.get_hamiltonian([kx, 0, 0])
    assert H_symb.shape == (3, 3), "Expected symbolic Hamiltonian shape (3,3)."
    # flattens the matrix to a list for checking
    for i, entry in enumerate(H_symb):
        # checks all entries of H for symbolic consistency
        assert isinstance(
            entry, sp.Basic), f"H_symb[{i}] is invalid: {repr(entry)} (type: {type(entry)})"
    assert H == H_symb, "Symbolic Hamiltonian should be equal to itself."

# ────────────────────────────────
# Basis Change
# ────────────────────────────────


def test_symbolic_basis_change_sanity():
    m, gamma, J, Mx = sp.symbols("m gamma J Mx")
    system = OrbitronicHamiltonianSystem(
        mass=m,
        orbital_texture_coupling=gamma,
        exchange_interaction_coupling=J,
        magnetisation=[Mx, 0, 0],
        symbolic=True
    )
    # this means no actual change
    system.set_basis(sp.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    L = system.L
    assert isinstance(
        L, list), "Expected angular momentum operators to be a list in symbolic mode."
    assert len(L) == 3, "Expected three angular momentum operators."
    for op in L:
        assert isinstance(
            op, sp.Matrix), "Expected each operator to be a sympy Matrix."


def test_numeric_basis_change_sanity():
    system = OrbitronicHamiltonianSystem(
        mass=1.0,
        orbital_texture_coupling=1.0,
        exchange_interaction_coupling=1.0,
        magnetisation=[1, 0, 0],
        symbolic=False
    )
    system.set_basis(np.eye(3))  # this means no actual change
    L = system.L
    assert isinstance(
        L, np.ndarray), "Expected angular momentum operators to be a numpy array in numeric mode."
    assert L.shape == (
        3, 3, 3), "Expected angular momentum operators shape (3, 3, 3)."


def test_numeric_unitary_transformation():
    """
    Tests the basis change method of the orbitronic system class in numeric mode
    with a specific unitary transformation as an example.
    """

    system = OrbitronicHamiltonianSystem(
        mass=1.0,
        orbital_texture_coupling=1.0,
        exchange_interaction_coupling=1.0,
        magnetisation=[1, 0, 0],
        symbolic=False
    )
    L = system.L
    sqrt2 = np.sqrt(2)
    # complex unitary matrix
    U_0 = np.array([[-1j/sqrt2, 0, 1j/sqrt2],
                   [1/sqrt2, 0, 1/sqrt2], [0, 1j, 0]])
    system.set_basis(U_0)
    L = system.L
    assert isinstance(
        L, np.ndarray), "Expected angular momentum operators to be a numpy array in numeric mode."
    assert L.shape == (
        3, 3, 3), "Expected angular momentum operators shape (3, 3, 3)."


def test_symbolic_unitary_transformation():
    m, gamma, J, Mx = sp.symbols("m gamma J Mx")
    system = OrbitronicHamiltonianSystem(
        mass=m,
        orbital_texture_coupling=gamma,
        exchange_interaction_coupling=J,
        magnetisation=[Mx, 0, 0],
        symbolic=True
    )
    L = system.L
    sqrt2 = sp.sqrt(2)  # it is crucial to use sympy's sqrt here
    U_0 = sp.Matrix([[-sp.I/sqrt2, 0, sp.I/sqrt2], [1/sqrt2, 0,
                    1/sqrt2], [0, sp.I, 0]])  # complex unitary matrix
    system.set_basis(U_0)
    L = system.L
    assert isinstance(
        L, list), "Expected angular momentum operators to be a list in symbolic mode."
    assert len(L) == 3, "Expected three angular momentum operators."
    for op in L:
        assert isinstance(
            op, sp.Matrix), "Expected each operator to be a sympy Matrix."


def test_unitary_transformation_both_modes():
    """
    Tests that a unitary transformation leads to the same angular momentum matrices
    in both symbolic and numeric modes.
    """

    # Numeric mode
    system_numeric = OrbitronicHamiltonianSystem(
        mass=1.0,
        orbital_texture_coupling=1.0,
        exchange_interaction_coupling=1.0,
        magnetisation=[1, 0, 0],
        symbolic=False
    )
    sqrt2 = np.sqrt(2)
    # complex unitary matrix
    U_0 = np.array([[-1j/sqrt2, 0, 1j/sqrt2],
                   [1/sqrt2, 0, 1/sqrt2], [0, 1j, 0]])
    system_numeric.set_basis(U_0)
    numeric_L = system_numeric.L  # after basis change

    # Symbolic mode
    m, gamma, J, Mx = sp.symbols("m gamma J Mx")
    system_symbolic = OrbitronicHamiltonianSystem(
        mass=m,
        orbital_texture_coupling=gamma,
        exchange_interaction_coupling=J,
        magnetisation=[Mx, 0, 0],
        symbolic=True
    )
    sqrt2 = sp.sqrt(2)  # it is crucial to use sympy's sqrt here
    U_0 = sp.Matrix([[-sp.I/sqrt2, 0, sp.I/sqrt2], [1/sqrt2, 0,
                    1/sqrt2], [0, sp.I, 0]])  # complex unitary matrix
    system_symbolic.set_basis(U_0)
    symbolic_L = system_symbolic.L  # after basis change

    # Convert symbolic to numeric for comparison
    symbolic_L_numeric = np.array(symbolic_L).astype(np.complex128)

    assert np.allclose(numeric_L, symbolic_L_numeric,
                       atol=1e-10), "Expected angular momentum operators to be the same in numeric and symbolic mode."

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
