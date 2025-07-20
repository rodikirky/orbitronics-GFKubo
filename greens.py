import numpy as np
import sympy as sp

class Greensfunctions:
    def __init__(self,
                 system,
                 energy_level,
                 infinitestimal,
                 retarded = True):
        self.system = system
        self.omega = energy_level
        self.eta = infinitestimal
        self.q = 1 if retarded else -1 # q= 1 for retarded, -1 for advanced

    def get_kspace_green(self, momentum):
        """Compute the Green's function in k-space."""
        k = momentum
        H = self.system.get_hamiltonian(momentum)
        identity = self.system.identity
        q = self.q # q is 1 for retarded, -1 for advanced
        omega = self.omega * identity # Ensure omega is a matrix for arithmatic operations
        imaginary_part = sp.I * self.eta * identity if self.system.symbolic else 1j * self.eta * identity

        if self.system.symbolic:
            # For symbolic systems, we use sympy's Matrix
            H = H.as_explicit()
            tobe_inverted = omega + q* imaginary_part - H
            tobe_inverted = sp.Matrix(tobe_inverted)
            assert tobe_inverted.det() != 0, "Hamiltonian must be invertible."
            GF_k = tobe_inverted.inv()
        else:
            # For numeric systems, we ensure H is a numpy array
            H = np.array(H)
            tobe_inverted = omega + q * imaginary_part - H
            assert np.linalg.det(tobe_inverted) != 0, "Hamiltonian must be invertible."
            GF_k = np.linalg.inv(tobe_inverted)
        return GF_k

    
