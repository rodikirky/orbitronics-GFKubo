from system_symbolic import OrbitronicHamiltonianSystem
import numpy as np
import sympy as sp

# ────────────────────────────────
# Basic Instantiation Tests 
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
    print("✅ Symbolic system created successfully.")

