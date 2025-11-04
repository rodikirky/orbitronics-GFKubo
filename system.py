from typing import Union, List, Optional
import numpy as np
import sympy as sp
from utils import is_unitary, sanitize_vector, get_identity, print_symbolic_matrix
import logging
import warnings

__all__ = ["OrbitronicHamiltonianSystem"]
log = logging.getLogger(__name__)


class OrbitronicHamiltonianSystem:
    def __init__(self,
                 # effective mass for the material
                 mass: Union[float, sp.Basic],
                 # kL coupling
                 orbital_texture_coupling: Union[float, sp.Basic],
                 # exchange coupling is zero for nonmagnets
                 exchange_interaction_coupling: Union[float, sp.Basic],
                 magnetisation: Union[List[Union[float, sp.Basic]], np.ndarray, sp.Matrix],
                 # default leads to canonical L matrices
                 basis: Optional[Union[np.ndarray, sp.Matrix]] = None):
        """
        OrbitronicHamiltonianSystem represents an orbitronic single-particle (non-interacting) quantum system 
        in either a symbolic or numeric mode with multiple angular momentum channels (L=1) represented
        by the standard 3x3 angular momentum matrices.

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
        """
        self.mass = mass
        if not self.mass > 0:
            raise ValueError("Mass must be positive.")
        self.gamma = orbital_texture_coupling
        self.J = exchange_interaction_coupling
        self.M = sanitize_vector(magnetisation, symbolic=True)
        self.identity = get_identity(3, symbolic=True)
        self.set_basis(self.identity if basis is None else basis)

        log.debug("Initialized %r", self) # developer snapshot

    def __repr__(self):
        try:
            I_shape = self.identity.shape
            return (f"{self.__class__.__name__}("
                    f"mass={self.mass}, I_shape={I_shape}, "
                    f"gamma={self.gamma}, J={self.J}, magnetisation={self.M})")
        except Exception:
            return f"{self.__class__.__name__}(unprintable; id=0x{id(self):x})"

    def set_basis(self, basis: Union[np.ndarray, sp.Matrix]) -> None:
        """Set angular momentum operators in the given basis."""
        Lx = sp.Matrix([[0, 0, 0], [0, 0, -sp.I], [0, sp.I, 0]])
        Ly = sp.Matrix([[0, 0, sp.I], [0, 0, 0], [-sp.I, 0, 0]])
        Lz = sp.Matrix([[0, -sp.I, 0], [sp.I, 0, 0], [0, 0, 0]])
        I = self.identity

        self.basis = sp.Matrix(basis) # basis change matrix

        # Determine whether the provided basis is the identity matrix
        is_identity = False
        is_identity = self.basis == I
        if is_identity:
            Ls = [Lx, Ly, Lz]
        else:
            U = self.basis

            assert is_unitary(
                U, symbolic=True), "Basis must be unitary"

            U_dagger = U.H
            Ls = [U_dagger @ L @ U for L in (Lx, Ly, Lz)]

            log.info("Basis transformation applied.")
            log.debug("Transformed L matrices:\nLx=%r\nLy=%r\nLz=%r", *Ls)

        self.L = Ls

    def get_potential(self, momentum: Union[np.ndarray, List, sp.Matrix]) -> Union[np.ndarray, sp.Matrix]:
        """Compute symbolic or numeric potential energy."""
        b = self.backend
        k = sanitize_vector(momentum, symbolic=True)
#
        dot_kL = sum((k[i] * self.L[i]
                        for i in range(3)), start=sp.zeros(3, 3))
        dot_ML = sum((self.M[i] * self.L[i]
                        for i in range(3)), start=sp.zeros(3, 3))

        orbital_term = self.gamma * (dot_kL @ dot_kL)
        exchange_term = self.J * dot_ML
        potential = orbital_term + exchange_term

        log.debug("Potential energy matrix successfully computed. %r", potential)

        return potential

    def get_hamiltonian(self, momentum: Union[np.ndarray, List, sp.Matrix]) -> Union[np.ndarray, sp.Matrix]:
        k = sanitize_vector(momentum, symbolic=True)
        kinetic = sum(k[i]**2 for i in range(3)) / (2 * self.mass)
        kinetic = kinetic * self.identity
        potential = self.get_potential(momentum)
        H = kinetic + potential
        log.debug("Total Hamiltonian successfully computed.")
        return H

class OrbitronicInterface:
    def __init__(self,
                 mass: Union[float, sp.Basic],
                 orbital_texture_coupling: Union[float, sp.Basic],
                 alpha: Union[float, sp.Basic],
                 beta: Union[float, sp.Basic],
                 crystal_field: Union[float, sp.Basic],
                 coeff: Union[float, sp.Basic], # normalization coefficient
                 # L matrices in chosen basis
                 angular_momentum: list[sp.Matrix, sp.Matrix, sp.Matrix]):
        self.m_int = mass
        self.gamma_int = orbital_texture_coupling
        self.alpha = alpha
        self.beta = beta
        self.delta_CF = crystal_field
        self.L_0 = coeff
        self.L = angular_momentum
        
        log.debug("Initialized %r", self) # developer snapshot

    def __repr__(self):
        try:
            return (f"{self.__class__.__name__}("
                    f"mass={self.m_int}, gamma={self.gamma_int}, L_0={self.L_0}, L_x={self.L[0]}"
                    f"alpha={self.alpha}, beta={self.beta}, delta_CF={self.delta_CF})")
        except Exception:
            return f"{self.__class__.__name__}(unprintable; id=0x{id(self):x})"
        # rest to follow

    def basis_change(self, basis: Union[np.ndarray, sp.Matrix]) -> None:
        L = self.L
        Lx, Ly, Lz = L
        identity = sp.eye(Lx.shape[0])
        if identity != basis:
            U = self.basis # basis change matrix
            assert is_unitary(
                U, symbolic=True), "Basis must be unitary"
            U_dagger = U.H
            Ls = [U_dagger @ l @ U for l in (Lx, Ly, Lz)]

            log.info("Basis transformation applied.")
            log.debug("Transformed L matrices:\nLx=%r\nLy=%r\nLz=%r", *Ls)

            self.L = Ls
        else:
            warnings.warn("Basis change attempted with the identity for a basis change matrix.")

    def orbital_texture_potential(self, k_x: Union[float, sp.Basic], k_y: Union[float, sp.Basic]):
        gamma = self.gamma_int
        L_x, L_y, _ = self.L
        dotkL = k_x * L_x + k_y * L_y
        V_tex = gamma * (dotkL @ dotkL)
        return V_tex
    
    def orbital_rashba_potential(self, k_x: Union[float, sp.Basic], k_y: Union[float, sp.Basic]):
        alpha = self.alpha
        L_x, L_y, _ = self.L    
        V_OR = alpha * (k_x * L_y - k_y * L_x)
        return V_OR
    
    def interfacial_hamiltonian(self, k_x: Union[float, sp.Basic], k_y: Union[float, sp.Basic]):
        m = self.m_int
        k_parallel_sqare = k_x**2 + k_y**2
        L_0 = self.L_0
        delta_CF = self.delta_CF
        beta = self.beta
        _, _, L_z = self.L
        L_z_sqare = L_z @ L_z

        kinetic_term = k_parallel_sqare / (2*m)
        V_tex = self.orbital_texture_potential(k_x,k_y)
        V_OR = self.orbital_rashba_potential(k_x,k_y)
        CF_term = (delta_CF + (beta * k_parallel_sqare)) * (1 - L_z_sqare)

        H_int = L_0 * (kinetic_term + V_tex + V_OR + CF_term)
        return H_int
