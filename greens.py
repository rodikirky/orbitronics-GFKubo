import numpy as np
import sympy as sp
from sympy import solveset, S, pprint
from typing import Callable, Union, Optional
from utils import invert_matrix, print_symbolic_matrix, sanitize_vector
import warnings


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
        A calculator for Green's functions.

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
        
        self.k_symbols = sp.symbols("k_x k_y k_z", real=True) 
        """
        Canonical momentum symbols used internally for solving:
        k_symbols[0] = k_x, k_symbols[1] = k_y, k_symbols[2] = k_z
        Only relevant for symbolic mode.
        """

        # for easy debugging along the way
        self.verbose = verbose

    def compute_kspace_greens_function(self, momentum: Union[np.ndarray, sp.Matrix]) -> Union[np.ndarray, sp.Matrix]:
        """
        Compute the Green's function for a single-particle Hamiltonian in momentum space by inverting
        (omega ± i*eta - H(k)).
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
            print("\nwith k = ", momentum)
            print_symbolic_matrix(H_k, name="H(k)") if self.symbolic else print("H(k) =\n", H_k)
            print_symbolic_matrix(tobe_inverted, name="( ω ± iη - H(k) )") if self.symbolic else print("Inversion target =\n", tobe_inverted)

        return invert_matrix(tobe_inverted, symbolic=self.symbolic)

    def compute_eigen_greens_inverse(self, momentum) -> Union[np.ndarray, sp.Matrix, list]:
        """
        Diagonalize the inverse Green's function matrix to obtain its eigenbasis and eigenvalues.
        Useful for identifying poles and simplifying root solving.

        Parameters:
        - momentum: value at which the Hamiltonian is evaluated

        Returns:
        - eigenbasis (matrix of eigenvectors) of G⁻¹(k)
        - eigenvalues of G⁻¹(k)
        - G⁻¹(k) diagonalized
        """
        symbolic = self.symbolic
        k = sanitize_vector(momentum, symbolic=symbolic)

        H_k = self.H(k)  # Hamiltonian at k
        omega_I = self.omega * self.I  # Frequency term scaled identity
        i_eta = (sp.I if self.symbolic else 1j) * self.eta * \
            self.I  # Imaginary part for broadening
        q = self.q  # q = 1 for retarded GF, q = -1 for advanced

        if self.symbolic:
            # Form the symbolic inverse Green's function
            G_inv = sp.Matrix(omega_I + q * i_eta - H_k)

            # Diagonalize: G⁻¹ = P D P⁻¹
            eigenbasis, G_inv_diag = G_inv.diagonalize()
            G_inv_diag = sp.Matrix(G_inv_diag)
            eigenvalues = G_inv_diag.diagonal()

            # Sanity check: construct diagonal matrix from original form by basis change
            G_inv_diagonalized = sp.simplify(invert_matrix(eigenbasis, symbolic=self.symbolic) @ G_inv @ eigenbasis)
            assert G_inv_diag[1] == G_inv_diagonalized[1], "Expected the diagonalized matrix to be diagonal."
            assert G_inv_diag[0].equals(G_inv_diagonalized[0]), "Expected the diagonalized matrix to have the eigenvalues on the diagonal."
        else:
            # Numerical case: use NumPy to obtain eigenvalues and eigenvectors
            G_inv = np.array(omega_I + q * i_eta - H_k)
            eigenvalues, eigenbasis = np.linalg.eig(G_inv)
            G_inv_diag = np.diag(eigenvalues) # Simply put the eigenvalues on the diagonal

            # Sanity check: construct diagonal matrix from original form by basis change
            assert np.allclose(G_inv_diag, invert_matrix(eigenbasis, symbolic=self.symbolic)@G_inv@eigenbasis, rtol=1e-10), \
                "Expected the diagonalized matrix to be diagonal."

        return eigenbasis, eigenvalues, G_inv_diag

    def compute_roots_greens_inverse(self, solve_for: Optional[int] = None):
        """
        Attempt to symbolically solve for the poles of the Green's function,
        i.e., values of momentum where one or more eigenvalues of the inverse Green's function vanish.
        Those poles correspond to the dispersion relations defining the band structure of the material.

       Parameters:
        - solve_for: Integer index (0 for k_x, 1 for k_y, 2 for k_z), or None to solve for all simultaneously.

        Returns:
        - List of (label, solution) tuples, where each label is "lambda_i=0" for the i-th eigenvalue,
          and solution is either:
            * a symbolic solution set (FiniteSet or ConditionSet), or
            * an error message if solving fails.
        """
        if solve_for is not None:
            if solve_for not in {0, 1, 2}:
                raise ValueError(
                    f"'solve_for' must be one of {{0, 1, 2}} "
                    f"corresponding to k_x, k_y, or k_z. Got: {solve_for}"
                )
            solve_for_symbol = self.k_symbols[solve_for]
        else:
            solve_for_symbol = None
        
        if not self.symbolic:
            warnings.warn("Root solving is only supported in symbolic mode. Enable symbolic=True.")
            return []

        # Define symbolic momentum components
        kx, ky, kz = self.k_symbols
        k = sp.Matrix([kx, ky, kz])

        # Compute eigenvalues of the inverse Green's function
        _, eigenvalues, _ = self.compute_eigen_greens_inverse(k)

        if self.verbose:
            print("\nDiagonal elements (eigenvalues) of G⁻¹(k):")
            for i, lambda_i in enumerate(eigenvalues):
                print(f"lambda_{i}(k):")
                pprint(lambda_i)

        root_solutions = []
        for i, lambda_i in enumerate(eigenvalues):
            lambda_i = sp.simplify(lambda_i)  # simplify for readability and solving
            if not lambda_i.is_polynomial(solve_for_symbol):
                warnings.warn(f"Solving λ_{i}(k) = 0 may fail: expression is not polynomial in {solve_for_symbol}.", stacklevel=2)


            try:
                # Solve λ_i(k) = 0 for specified variable or for full vector
                variable_to_solve = solve_for_symbol if solve_for_symbol is not None else (kx, ky, kz)
                sol = solveset(sp.Eq(lambda_i, 0), variable_to_solve, domain=S.Reals)
                root_solutions.append((f"lambda_{i}=0", sol))
            except Exception as e:
                root_solutions.append((f"lambda_{i}=0", f"Error during solving: {e}"))

        return root_solutions
    
    def compute_rspace_greens_function(self):
        k = 1
        G_k = self.compute_kspace_greens_function(k)
        fourier_transform = G_k

        return fourier_transform
