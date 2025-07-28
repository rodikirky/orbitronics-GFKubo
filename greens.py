import numpy as np
import sympy as sp
from typing import Callable, Union
from utils import invert_matrix, print_symbolic_matrix, sanitize_vector


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
        i_eta = (sp.I if self.symbolic else 1j) * self.eta * \
            self.I  # Imaginary part for broadening
        q = self.q  # q = 1 for retarded GF, q = -1 for advanced

        tobe_inverted = omega_I + q * i_eta - H_k

        if self.symbolic:
            # Ensure symbolic matrix for symbolic inversion
            tobe_inverted = sp.Matrix(tobe_inverted)
        else:
            # Ensure numeric matrix for numeric inversion
            tobe_inverted = np.array(tobe_inverted)

        if self.verbose:
            print("\nComputing Green's function at k:")
            print_symbolic_matrix(
                H_k, name="H(k)") if self.symbolic else print("H(k) =\n", H_k)
            print_symbolic_matrix(tobe_inverted, name="( ω ± iη - H(k) )") if self.symbolic else print(
                "Inversion target =\n", tobe_inverted)

        return invert_matrix(tobe_inverted, symbolic=self.symbolic)

    def compute_eigenbasis_greens_invert(self, momentum) -> Union[np.ndarray, sp.Matrix, list]:
        """
        Compute the basis which diagonalizes G⁻¹(k, ω) = (ω + iη)I - H(k).  
        Diagonalize G⁻¹(k, ω) with this basis.

        Parameters:
        - NONE

        Returns:
        - Basis change matrix that diagonalizes G⁻¹(k, ω).
        - Diagonalized amtrix G⁻¹(k, ω)
        """
        symbolic = self.symbolic
        k = sanitize_vector(momentum, symbolic=symbolic)

        H_k = self.H(k)  # Hamiltonian at k
        omega_I = self.omega * self.I  # Frequency term scaled identity
        i_eta = (sp.I if self.symbolic else 1j) * self.eta * \
            self.I  # Imaginary part for broadening
        q = self.q  # q = 1 for retarded GF, q = -1 for advanced

        if self.symbolic:
            # denominator of G_k as a matrix depending on k
            G_inv = sp.Matrix(omega_I + q * i_eta - H_k)
            eigenbasis, G_inv_diag = G_inv.diagonalize()
            G_inv_diag = sp.Matrix(G_inv_diag)
            assert G_inv_diag == sp.simplify(invert_matrix(
                eigenbasis) @ G_inv @ eigenbasis), "Expected the " \
                "diagonalized denominator to be diagonal with the eigenvalues on the diagonal."
            eigenvalues = G_inv_diag.diagonal()
        else:
            # denominator of G_k as a matrix depending on k
            G_inv = np.array(omega_I + q * i_eta - H_k)
            eigenvalues, eigenbasis = np.linalg.eig(G_inv)
            G_inv_diag = np.diag(eigenvalues)
            assert np.allclose(G_inv_diag, invert_matrix(eigenbasis)@G_inv@eigenbasis, rtol=1e-10,
                               err_msg="Expected the diagonalized denominator to be diagonal with " \
                               "the eigenvalues on the diagonal.")

        return eigenbasis, eigenvalues, G_inv_diag

    def compute_roots_greens_invert(self) -> Union[np.ndarray, sp.Matrix, list]:
        """
        Compute the Green's function's poles, i.e. the momenta which 
        make (ω + iη)I - H(k) non-invertible, in k-space.
        We find those poles n k-space by computing the determinant 
        and solving for k, such that det{(ω + iη)I - H(k)} = 0.

        Or we diagonalize (ω + iη)I - H(k). We should get three eigenvalues 
        that are second order polynomials in k, such that there are two (possibly different) 
        values for each k_x, k_y and k_z such that one or more of the eigenvalues 
        vanish, resulting in a non-invertible matrix with det{(ω + iη)I - H(k)} = 0.

        Parameters:
        - NONE

        Returns:
        - List of momenta k as np.ndarray or sp.matrix
        """

        k_roots = []

        if self.symbolic:
            kx, ky, kz = sp.symbols("k_x k_y k_z")
            k = sp.Matrix([kx, ky, kz])
            eigenbasis, eigenvalues, G_inv_diag = self.compute_eigenbasis_greens_invert(k)
            
            # now, we need to find the k values, such that one or more eigenvalues vanish
        else:
            print("Examination of the roots makes little sense in numeric mode." \
                  "Switch to symbolic mode to gain more insight!")
        return k_roots
