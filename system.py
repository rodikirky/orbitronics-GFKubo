import numpy as np

class HamiltonianSystem:
    def __init__(self,
                 mass: float,
                 orbital_texture_coupling: float,
                 exchange_interaction_coupling: float,
                 magnetisation: np.ndarray,
                 basis: np.ndarray = np.eye(3)):
        """
        Initialize the system with physical parameters and an optional basis.

        Parameters:
        - mass: Particle mass
        - orbital_texture_coupling: γ coupling constant
        - exchange_interaction_coupling: J coupling constant
        - magnetisation: M vector
        - basis: 3x3 matrix defining the angular momentum basis, defaults to identity matrix
        """
        self.mass = mass
        self.gamma = orbital_texture_coupling
        self.J = exchange_interaction_coupling
        self.M = magnetisation
        self.set_basis(basis)

    def set_basis(self, basis: np.ndarray):
        """Set and cache the angular momentum operators in a given basis."""
        self.basis = basis
        # Canonical L matrices in the default basis
        Lx = np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]])
        Ly = np.array([[0, 0, 1j], [0, 0, 0], [-1j, 0, 0]])
        Lz = np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]])
        if np.allclose(basis, np.eye(3)):
            self.L = np.array([Lx, Ly, Lz])
        else:
            U = basis
            self.L = np.array([U @ Lx @ U.T, U @ Ly @ U.T, U @ Lz @ U.T])

    def change_basis(self, new_basis: np.ndarray):
        """Switch to a new basis and update angular momentum matrices."""
        self.set_basis(new_basis)
    
    def get_potential(self, momentum: np.ndarray) -> np.ndarray:
        """
        Calculate the potential energy matrix at a given momentum.

        Parameters:
        - momentum: k vector

        Returns:
        - potential: 3x3 Hermitian matrix
        - This result automatically updates after a basis change, when called again.
        """
        orbital_term = self.gamma * (np.dot(momentum, self.L)) ** 2
        exchange_term = self.J * np.dot(self.M, self.L)
        return orbital_term + exchange_term

    def get_hamiltonian(self, momentum: np.ndarray) -> np.ndarray:
        """
        Return the full Hamiltonian matrix.

        Parameters:
        - momentum: k vector

        Returns:
        - Hamiltonian: 3x3 Hermitian matrix
        - This result automatically updates after a basis change, when called again.
        """
        kinetic_term = np.dot(momentum, momentum) / (2 * self.mass)
        potential_term = self.get_potential(momentum)
        return kinetic_term + potential_term
