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

        # Set symbolic mode
        self.symbolic = symbolic
        self.backend = sp if symbolic else np

        # Safe assignment of scalars
        self.mass = sp.sympify(mass) if symbolic else float(mass)
        self.gamma = sp.sympify(orbital_texture_coupling) if symbolic else float(orbital_texture_coupling)
        self.J = sp.sympify(exchange_interaction_coupling) if symbolic else float(exchange_interaction_coupling)

        # Safe assignment of magnetisation
        self.M = self._sanitize_vector(magnetisation)

        default_eye = sp.eye(3) if symbolic else np.eye(3)
        self.set_basis(default_eye if basis is None else basis)

    def _sanitize_vector(self, v):
        if self.symbolic:
            if isinstance(v, np.ndarray):
                return [sp.sympify(val) for val in v]
            elif isinstance(v, sp.Matrix):
                return list(v)
            else:
                return v  # assume already symbolic list
        else:
            return np.asarray(v, dtype=np.float64)

    def set_basis(self, basis):
        """Set angular momentum operators in the given basis."""
        b = self.backend
        Lx = b.Matrix([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]])
        Ly = b.Matrix([[0, 0, 1j], [0, 0, 0], [-1j, 0, 0]])
        Lz = b.Matrix([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]])
        I = b.eye(3)

        self.basis = b.Matrix(basis)

        is_identity = False
        if self.symbolic:
            is_identity = self.basis == I
        else:
            is_identity = np.allclose(np.array(self.basis).astype(np.float64), np.eye(3))

        if is_identity:
            Ls = [Lx, Ly, Lz]
        else:
            U = self.basis
            Ls = [U @ L @ U.T for L in (Lx, Ly, Lz)]

        if self.symbolic:
            self.L = Ls  # plain list
        else:
            self.L = np.stack(Ls, axis=0)  # 3D array: shape (3, 3, 3)

    def get_potential(self, momentum):
        """Compute symbolic or numeric potential energy."""
        b = self.backend
        k = self._sanitize_vector(momentum)
        
        if self.symbolic:
            dot_kL = sum(k[i] * self.L[i] for i in range(3))
            dot_ML = sum(self.M[i] * self.L[i] for i in range(3))
        else:
            dot_kL = np.tensordot(k, self.L, axes=1)
            dot_ML = np.tensordot(self.M, self.L, axes=1)

        orbital_term = self.gamma * (dot_kL @ dot_kL)
        exchange_term = self.J * dot_ML
        return orbital_term + exchange_term


    def get_hamiltonian(self, momentum):
        """Return symbolic or numeric Hamiltonian."""
        k = self._sanitize_vector(momentum)
        kinetic = sum(momentum[i]**2 for i in range(3)) / (2 * self.mass)
        return kinetic + self.get_potential(momentum)

    def get_symbolic_hamiltonian(self):
        """Convenience method to return Hamiltonian with default symbols."""
        if not self.symbolic:
            raise ValueError("Hamiltonian is not symbolic. Set symbolic=True at init.")
        kx, ky, kz = sp.symbols("k_x k_y k_z", real=True)
        momentum = [kx, ky, kz]
        return self.get_hamiltonian(momentum)
