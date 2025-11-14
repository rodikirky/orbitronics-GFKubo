from greens import GreensFunctionCalculator
import sympy as sp

# region Hamiltonians
## Constant Hamiltonians (ignore k, test N×N size only):
def constant_hamiltonian_3band(k_vec):
    """3x3 constant Hamiltonian."""
    return sp.eye(3)
def constant_hamiltonian_2band(k_vec): 
    """2x2 constant Hamiltonian."""
    return sp.eye(2)
def constant_hamiltonian_1band(k_vec):
    """1x1 constant Hamiltonian."""
    return sp.eye(1)
## Numeric:

## Diagonal "kinetic" Hamiltonians (dimension-aware in k-space):
def kinetic_hamiltonian_3D(k_vec):
    """3D k-space, 3x3 band."""
    kx, ky, kz = k_vec
    H_k = kx**2 + ky**2 + kz**2
    return H_k * sp.eye(3)
def kinetic_hamiltonian_2D(k_vec):
    """2D k-space, 2x2 band."""
    kx, ky = k_vec
    H_k = kx**2 + ky**2
    return H_k * sp.eye(2)
def kinetic_hamiltonian_1D_scalar(k):
    """
    1D k-space, 1x1 band.
    For d=1 the class passes a scalar to H(k), so we accept a scalar,
    and return a scalar (to trigger the d==1 code path).
    """
    return k**2  # will become a 1×1 Matrix via sp.Matrix([H(k)])

## Non-square Hamiltonian for error testing:
def nonsquare_hamiltonian(k_vec):
    return sp.Matrix([[1, 2, 3],
                      [4, 5, 6]])  # 2×3, not square
# endregion

# ────────────────────────────────
# region Init, repr, str
# ────────────────────────────────
def test_initiate_calculator_3D():
    calc = GreensFunctionCalculator(constant_hamiltonian_3band)
    assert callable(calc.H)
    assert calc.d == 3
    assert isinstance(calc.k_symbols, tuple)
    assert isinstance(calc.k_vec, sp.Matrix)
    assert calc.H_k == sp.eye(3)
    assert calc.N == 3
    assert calc.I == calc.H_k
    assert calc.q == 1
    assert (isinstance(calc.omega, sp.Symbol) and isinstance(calc.eta, sp.Symbol))
    assert calc.green_type == "retarded (+iη)"

def test_initiate_calculator_2D():
    calc = GreensFunctionCalculator(constant_hamiltonian_2band)
    assert callable(calc.H)
    assert calc.d == 3
    assert isinstance(calc.k_symbols, tuple)
    assert isinstance(calc.k_vec, sp.Matrix)
    assert calc.H_k == sp.eye(2)
    assert calc.N == 2
    assert calc.I == calc.H_k
    assert calc.q == 1
    assert (isinstance(calc.omega, sp.Symbol) and isinstance(calc.eta, sp.Symbol))
    assert calc.green_type == "retarded (+iη)"

def test_initiate_calculator_1D():
    calc = GreensFunctionCalculator(constant_hamiltonian_1band)
    assert callable(calc.H)
    assert calc.d == 3
    assert isinstance(calc.k_symbols, tuple)
    assert isinstance(calc.k_vec, sp.Matrix)
    assert calc.H_k == sp.eye(1)
    assert calc.N == 1
    assert calc.I == calc.H_k
    assert calc.q == 1
    assert (isinstance(calc.omega, sp.Symbol) and isinstance(calc.eta, sp.Symbol))
    assert calc.green_type == "retarded (+iη)"

def test_initiate_advanced_3D():
    calc = GreensFunctionCalculator(constant_hamiltonian_3band, retarded=False)
    assert callable(calc.H)
    assert calc.d == 3
    assert isinstance(calc.k_symbols, tuple)
    assert isinstance(calc.k_vec, sp.Matrix)
    assert calc.H_k == sp.eye(3)
    assert calc.N == 3
    assert calc.I == calc.H_k
    assert calc.q == -1
    assert (isinstance(calc.omega, sp.Symbol) and isinstance(calc.eta, sp.Symbol))
    assert calc.green_type == "advanced (-iη)"

# add tests fpr repr and str here

# endregion

# ────────────────────────────────
# region Matrix construct
# ────────────────────────────────
def test_greens_inverse_constant_3D():
    calc = GreensFunctionCalculator(constant_hamiltonian_3band)
    G_inv = calc.greens_inverse()
    assert G_inv.shape[0] == calc.N
    assert G_inv.free_symbols
    assert G_inv[0,1] == 0

def test_greens_inverse_diag_3D():
    calc = GreensFunctionCalculator(kinetic_hamiltonian_3D)
    G_inv = calc.greens_inverse()
    assert G_inv.shape[0] == calc.N
    assert G_inv.free_symbols & set(calc.k_vec)
    assert G_inv[0,1] == 0

def test_greens_inverse_diag_2D():
    calc = GreensFunctionCalculator(kinetic_hamiltonian_2D)
    G_inv = calc.greens_inverse()
    assert G_inv.shape[0] == calc.N
    assert G_inv.free_symbols & set(calc.k_vec)
    assert G_inv[0,1] == 0

def test_greens_inverse_diag_1D():
    calc = GreensFunctionCalculator(kinetic_hamiltonian_1D_scalar)
    G_inv = calc.greens_inverse()
    assert G_inv.shape[0] == calc.N
    assert G_inv.free_symbols & set(calc.k_vec)

# endregion

# ────────────────────────────────
# region Inversion
# ────────────────────────────────

# endregion