import numpy as np
import sympy as sp
from typing import Callable, Union
from utils import invert_matrix, print_symbolic_matrix


class GreensFunctionCalculator:
    def __init__(self,
                 hamiltonian: Callable[[Union[list, np.ndarray, sp.Matrix]], Union[np.ndarray, sp.Matrix]],
                 identity: Union[np.ndarray, sp.Matrix],
                 symbolic: bool,
                 # omega
                 energy_level: Union[float, sp.Basic],
                 # eta
                 infinitestimal: float,
                 # defaults to retarded Green's functions
                 retarded: bool = True,
                 # defaults to non-verbose
                 # if verbose=True, intermediate steps will be printed out
                 verbose: bool = False):
        """
        A calculator for Green's functions in momentum space.

        Parameters:
        - hamiltonian: a function that takes momentum k and returns the Hamiltonian matrix
        - identity: identity matrix (3x3) for the appropriate backend
        - symbolic: whether to use symbolic (sympy) or numeric (numpy) mode
        - energy_level: scalar ω
        - infinitestimal: small η > 0 to define the imaginary part
        - retarded: if True computes retarded Green's function; else advanced
        - verbose: if True, prints intermediate matrix states for debugging
        """

        self.H = hamiltonian
        self.I = identity
        self.symbolic = symbolic
        self.omega = energy_level
        self.eta = infinitestimal
        self.q = 1 if retarded else -1

        # for easy debugging along the way
        self.verbose = verbose

    def compute_kspace_greens_function(self, momentum: Union[np.ndarray, sp.Matrix]) -> Union[np.ndarray, sp.Matrix]:
        """
        Compute the Green's function G(k, ω) = [(ω + iη)I - H(k)]⁻¹.

        Parameters:
        - momentum: 3-vector for k-space input

        Returns:
        - Green's function matrix at momentum k
        """
        H_k = self.H(momentum)  # Hamiltonian at k
        omega_I = self.omega * self.I  # Frequency term scaled identity
        i_eta = (sp.I if self.symbolic else 1j) * self.eta * self.I  # Imaginary part for broadening
        q = self.q # q = 1 for retarded GF, q = -1 for advanced

        tobe_inverted = omega_I + q * i_eta - H_k

        if self.symbolic:
            # Ensure symbolic matrix for symbolic inversion
            tobe_inverted = sp.Matrix(tobe_inverted)
        else:
            # Ensure numeric matrix for numeric inversion
            tobe_inverted = np.array(tobe_inverted)

        if self.verbose:
            print("\nComputing Green's function at k:")
            print_symbolic_matrix(H_k, name="H(k)") if self.symbolic else print("H(k) =\n", H_k)
            print_symbolic_matrix(tobe_inverted, name="( ω ± iη - H(k) )") if self.symbolic else print("Inversion target =\n", tobe_inverted)

        return invert_matrix(tobe_inverted, symbolic=self.symbolic)
