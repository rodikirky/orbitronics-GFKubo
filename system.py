from typing import Union, List, Optional
import numpy as np
import sympy as sp
from utils import is_unitary, sanitize_vector, get_identity


class OrbitronicHamiltonianSystem:
    """
    OrbitronicHamiltonianSystem represents an orbitronic quantum system 
    in either a symbolic or numeric mode.

    Supports computation of angular momentum-coupled potentials and Hamiltonians
    in a specified basis. Accepts both symbolic expressions (via SymPy) and 
    numerical arrays (via NumPy), making it suitable for both analytical modeling 
    and simulation.

    Parameters:
    - mass: Scalar mass (numeric or symbolic)
    - orbital_texture_coupling: Coupling strength γ
    - exchange_interaction_coupling: Coupling strength J
    - magnetisation: 3D vector representing magnetic moment
    - basis: Optional 3×3 matrix defining basis transformation
    - symbolic: Whether to use symbolic backend (SymPy) or numeric (NumPy)
    """

    def __init__(self,
                 # effective mass for the material
                 mass: Union[float, sp.Basic],
                 # kL coupling
                 orbital_texture_coupling: Union[float, sp.Basic],
                 # zero for nonmagnets
                 exchange_interaction_coupling: Union[float, sp.Basic],
                 magnetisation: Union[List[Union[float, sp.Basic]], np.ndarray, sp.Matrix],
                 # default leads to canonical L matrices
                 basis: Optional[Union[np.ndarray, sp.Matrix]] = None,
                 symbolic: bool = False):  # defaults to numeric mode

        def _is_symbolic(val):
            return isinstance(val, sp.Basic)

        # Check consistency if symbolic=False
        if not symbolic:
            scalar_params = {
                "mass": mass,
                "orbital_texture_coupling": orbital_texture_coupling,
                "exchange_interaction_coupling": exchange_interaction_coupling
            }
            for name, val in scalar_params.items():
                if _is_symbolic(val):
                    raise TypeError(
                        f"Cannot use symbolic value for '{name}' when symbolic=False.\n"
                        f"Hint: Set symbolic=True if you want to use symbols like '{val}'."
                    )

            if isinstance(magnetisation, (list, tuple, np.ndarray)):
                for i, val in enumerate(magnetisation):
                    if _is_symbolic(val):
                        raise TypeError(
                            f"Cannot use symbolic value for 'magnetisation[{i}]' when symbolic=False.\n"
                            f"Hint: Set symbolic=True if you want to use symbols like '{val}'."
                        )

        # Set symbolic/numeric mode
        self.symbolic = symbolic
        self.backend = sp if symbolic else np

        # Safe assignment of scalars
        self.mass = sp.sympify(mass) if symbolic else float(mass)
        self.gamma = sp.sympify(orbital_texture_coupling) if symbolic else float(
            orbital_texture_coupling)
        self.J = sp.sympify(exchange_interaction_coupling) if symbolic else float(
            exchange_interaction_coupling)

        self.M = sanitize_vector(magnetisation, symbolic)

        self.identity = get_identity(3, symbolic)
        self.make_matrix = sp.Matrix if symbolic else np.array
        self.set_basis(self.identity if basis is None else basis)

    def set_basis(self, basis: Union[np.ndarray, sp.Matrix]) -> None:
        """Set angular momentum operators in the given basis."""
        b = self.backend
        Lx = self.make_matrix([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]])
        Ly = self.make_matrix([[0, 0, 1j], [0, 0, 0], [-1j, 0, 0]])
        Lz = self.make_matrix([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]])
        I = self.identity

        self.basis = self.make_matrix(basis)

        # Determine whether the provided basis is the identity matrix
        is_identity = False
        if self.symbolic:
            is_identity = self.basis == I
        else:
            if np.iscomplex(self.basis).any():
                is_identity = False
            else:
                is_identity = np.allclose(
                    np.array(self.basis).astype(np.float64), np.eye(3))

        if is_identity:
            Ls = [Lx, Ly, Lz]
        else:
            U = self.basis

            assert is_unitary(
                U, symbolic=self.symbolic), "Basis must be unitary"

            U_dagger = U.H if self.symbolic else np.linalg.inv(U)
            Ls = [U_dagger @ L @ U for L in (Lx, Ly, Lz)]

        self.L = Ls if self.symbolic else np.stack(Ls, axis=0)

    def get_potential(self, momentum: Union[np.ndarray, List, sp.Matrix]) -> Union[np.ndarray, sp.Matrix]:
        """Compute symbolic or numeric potential energy."""
        b = self.backend
        k = sanitize_vector(momentum, symbolic=self.symbolic)

        if self.symbolic:
            dot_kL = sum((k[i] * self.L[i]
                         for i in range(3)), start=sp.zeros(3, 3))
            dot_ML = sum((self.M[i] * self.L[i]
                         for i in range(3)), start=sp.zeros(3, 3))
        else:
            dot_kL = np.tensordot(k, self.L, axes=1)
            dot_ML = np.tensordot(self.M, self.L, axes=1)

        orbital_term = self.gamma * (dot_kL @ dot_kL)
        exchange_term = self.J * dot_ML
        return orbital_term + exchange_term

    def get_hamiltonian(self, momentum: Union[np.ndarray, List, sp.Matrix]) -> Union[np.ndarray, sp.Matrix]:
        k = sanitize_vector(momentum, symbolic=self.symbolic)
        kinetic = sum(k[i]**2 for i in range(3)) / (2 * self.mass)
        kinetic = kinetic * self.identity
        return kinetic + self.get_potential(momentum)

    def get_symbolic_hamiltonian(self) -> sp.Matrix:
        """Convenience method to return Hamiltonian with default symbols."""
        if not self.symbolic:
            raise ValueError(
                "Hamiltonian is not symbolic. Set symbolic=True at init.")
        kx, ky, kz = sp.symbols("k_x k_y k_z", real=True)
        momentum = [kx, ky, kz]
        return self.get_hamiltonian(momentum)
