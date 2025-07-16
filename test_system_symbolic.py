from system_symbolic import OrbitronicHamiltonianSystem
import numpy as np
import sympy as sp

def test_create_numeric_system():
    system = OrbitronicHamiltonianSystem(
        mass=1.0,
        orbital_texture_coupling=1.0,
        exchange_interaction_coupling=1.0,
        magnetisation=[1, 0, 0],
        symbolic=False
    )
    print("✅ Numeric system created successfully.")

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

test_create_numeric_system()
#test_create_symbolic_system()