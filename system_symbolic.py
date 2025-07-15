import numpy as np
import sympy as sp

class HamiltonianSystem:
    """
    Initialize the system with physical parameters and an optional basis.

    Parameters:
       - mass, gamma, J, M: Can be numeric or symbolic
       - basis: 3x3 matrix; if None, uses identity matrix
       - symbolic: If True, use sympy instead of numpy
    """
    def __init__(self,
             mass,
             orbital_texture_coupling,
             exchange_interaction_coupling,
             magnetisation,
             basis=None,
             symbolic=False):
         self.symbolic = symbolic
         self.backend = sp if symbolic else np
         self.mass = mass
         self.gamma = orbital_texture_coupling
         self.J = exchange_interaction_coupling
         self.M = magnetisation
         
         default_eye = sp.eye(3) if symbolic else np.eye(3)
         self.set_basis(default_eye if basis is None else basis)

    def set_basis(self, basis):
        """Set angular momentum operators in the given basis."""
        b = self.backend
        I = b.eye(3)
        Lx = b.Matrix([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]])
        Ly = b.Matrix([[0, 0, 1j], [0, 0, 0], [-1j, 0, 0]])
        Lz = b.Matrix([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]])

        self.basis = b.Matrix(basis)
        if self.symbolic or np.allclose(np.array(self.basis).astype(np.float64), np.eye(3)):
            self.L = [Lx, Ly, Lz]
        else:
            U = self.basis
            self.L = [U @ L @ U.T for L in (Lx, Ly, Lz)]

    def get_potential(self, momentum):
        """Compute symbolic or numeric potential energy."""
        k = momentum
        b = self.backend
        dot_kL = sum(k[i] * self.L[i] for i in range(3))
        orbital_term = self.gamma * (dot_kL ** 2)
        exchange_term = self.J * sum(self.M[i] * self.L[i] for i in range(3))
        return orbital_term + exchange_term

    def get_hamiltonian(self, momentum):
        """Return symbolic or numeric Hamiltonian."""
        kinetic = sum(momentum[i]**2 for i in range(3)) / (2 * self.mass)
        return kinetic + self.get_potential(momentum)

    def get_symbolic_hamiltonian(self):
        """Convenience method to return Hamiltonian with default symbols."""
        if not self.symbolic:
            raise ValueError("Hamiltonian is not symbolic. Set symbolic=True at init.")
        kx, ky, kz = sp.symbols("k_x k_y k_z", real=True)
        momentum = [kx, ky, kz]
        return self.get_hamiltonian(momentum)
