from system_symbolic import OrbitronicHamiltonianSystem
import numpy as np
import sympy as sp

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
    assert isinstance(system.mass, float)
    assert isinstance(system.gamma, float)
    assert isinstance(system.J, float)

    assert isinstance(system.M, np.ndarray)
    assert system.M.shape == (3,)
    assert np.allclose(system.M, [1, 0, 0])

    assert isinstance(system.basis, np.ndarray)
    assert system.basis.shape == (3, 3)
    assert np.allclose(system.basis, np.eye(3))

    assert system.symbolic is False

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
    assert system.M.shape == (3, 1)
    assert system.M[0] == Mx
    assert system.M[1] == 0
    assert system.M[2] == 0

    assert isinstance(system.basis, sp.Matrix)
    assert system.basis.shape == (3, 3)
    assert system.basis == sp.eye(3)

    assert system.symbolic is True

# ────────────────────────────────
# Hamiltonian Construction
# ────────────────────────────────

def test_numeric_hamiltonian_shape():
    system = OrbitronicHamiltonianSystem(1.0, 1.0, 0.0, [0, 0, 0], symbolic=False)
    H = system.get_hamiltonian([1.0, 0.0, 0.0])
    assert H.shape == (3, 3)
    assert isinstance(H, np.ndarray)

def test_symbolic_hamiltonian_entries():
    m, gamma, J = sp.symbols("m gamma J")
    system = OrbitronicHamiltonianSystem(m, gamma, J, [0, 0, 0], symbolic=True)
    H = system.get_symbolic_hamiltonian()
    assert H.shape == (3, 3)
    assert isinstance(H[0, 0], sp.Basic)