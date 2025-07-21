import numpy as np
import sympy as sp
from typing import Callable, Union
from utils import invert_matrix


class GreensFunctionCalculator:
    def __init__(self,
                 hamiltonian: Callable[[Union[list, np.ndarray, sp.Matrix]], Union[np.ndarray, sp.Matrix]],
                 identity: Union[np.ndarray, sp.Matrix],
                 symbolic: bool,
                 energy_level: Union[float, sp.Basic],
                 infinitestimal: float,
                 retarded: bool = True):
        """
        A calculator for Green's functions in momentum space.

        Parameters:
        - hamiltonian: a function that takes momentum k and returns the Hamiltonian matrix
        - identity: identity matrix (3x3) for the appropriate backend
        - symbolic: whether to use symbolic (sympy) or numeric (numpy) mode
        - energy_level: scalar ω
        - infinitestimal: small η > 0 to define the imaginary part
        - retarded: if True computes retarded Green's function; else advanced
        """
        self.H = hamiltonian
        self.I = identity
        self.symbolic = symbolic
        self.omega = energy_level
        self.eta = infinitestimal
        self.q = 1 if retarded else -1

    def compute_kspace_greens_function(self, momentum: Union[np.ndarray, sp.Matrix]) -> Union[np.ndarray, sp.Matrix]:
        """
        Compute the Green's function G(k, ω) = [(ω + iη)I - H(k)]⁻¹.

        Parameters:
        - momentum: 3-vector for k-space input

        Returns:
        - Green's function matrix at momentum k
        """
        H_k = self.H(momentum)  # hamiltonian in k-space
        omega_I = self.omega * self.I  # allows for matric arithmetic with a scalar
        i_eta = (sp.I if self.symbolic else 1j) * \
            self.eta * self.I  # imaginary part
        q = self.q  # retarded (q = 1) or advanced (q = -1)
        tobe_inverted = omega_I + q * i_eta - H_k

        if self.symbolic:
            # Ensure full matrix for symbolic inversion
            tobe_inverted = sp.Matrix(tobe_inverted.as_explicit())
        else:
            tobe_inverted = np.array(tobe_inverted)

        return invert_matrix(tobe_inverted, symbolic=self.symbolic)
