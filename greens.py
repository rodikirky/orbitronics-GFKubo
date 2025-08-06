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
                 dimension: int = 3,
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

        # Choice of dimension determines default momentum symbols:
        self.d = dimension
        if self.d == 1:
            self.k_symbols = sp.symbols("k", real=True)
        elif self.d in {1,2}:
            self.k_symbols = sp.symbols(" ".join(f"k_{d}" for d in "xyz"[:self.d]), real=True)
        else:
            warnings.warn(
                "Class support only 1D, 2D and 3D computation. Choose dimension from {1,2,3}.")

        """
        Canonical momentum symbols used internally for solving:
        k_symbols[0] = "k_x", k_symbols[1] = "k_y", k_symbols[2] = "k_z"
        Only relevant for symbolic mode.
        """

        # for easy debugging along the way
        self.verbose = verbose

    def info(self):
        """
        Print a summary of the internal configuration of this calculator instance.
        """
        print("\nGreensFunctionCalculator configuration:")
        print("======================================")
        print(f"Symbolic mode         : {self.symbolic}")
        print(f"Verbose mode          : {self.verbose}")
        print(f"Energy ω              : {self.omega}")
        print(f"Infinitesimal η       : {self.eta}")
        green_type = "retarded (+iη)" if self.q == 1 else "advanced (−iη)"
        print(f"Green's function type : {green_type}")
        print(f"Identity matrix       : {self.I.shape}")
        print(f"H(k) callable         : {'Yes' if callable(self.H) else 'No'}")
        print(f"Momentum symbols      : {self.k_symbols}")
        print("======================================\n")

    # --- GF computation in k-space ---

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
            print_symbolic_matrix(
                H_k, name="H(k)") if self.symbolic else print("H(k) =\n", H_k)
            print_symbolic_matrix(tobe_inverted, name="( ω ± iη - H(k) )") if self.symbolic else print(
                "Inversion target =\n", tobe_inverted)

        return invert_matrix(tobe_inverted, symbolic=self.symbolic)

    # --- Analytical helpers ---

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
            G_inv_diagonalized = sp.simplify(invert_matrix(
                eigenbasis, symbolic=self.symbolic) @ G_inv @ eigenbasis)
            assert G_inv_diag.equals(
                G_inv_diagonalized), "Expected diagonalized matrix to be diagonal and to have the eigenvalues on the diagonal."

        else:
            # Numerical case: use NumPy to obtain eigenvalues and eigenvectors
            G_inv = np.array(omega_I + q * i_eta - H_k)
            eigenvalues, eigenbasis = np.linalg.eig(G_inv)
            # Simply put the eigenvalues on the diagonal
            G_inv_diag = np.diag(eigenvalues)

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
            warnings.warn(
                "Root solving is only supported in symbolic mode. Enable symbolic=True.")
            return []

        # Define symbolic momentum components
        kx, ky, kz = self.k_symbols
        k = self.k_symbols

        # Compute eigenvalues of the inverse Green's function
        _, eigenvalues, _ = self.compute_eigen_greens_inverse(k)

        if self.verbose:
            print("\nDiagonal elements (eigenvalues) of G⁻¹(k):")
            for i, lambda_i in enumerate(eigenvalues):
                print(f"lambda_{i}(k):")
                pprint(lambda_i)

        root_solutions = []
        for i, lambda_i in enumerate(eigenvalues):
            # simplify for readability and solving
            lambda_i = sp.simplify(lambda_i)
            if not lambda_i.is_polynomial(solve_for_symbol):
                warnings.warn(
                    f"Solving λ_{i}(k) = 0 may fail: expression is not polynomial in {solve_for_symbol}.", stacklevel=2)

            try:
                # Solve λ_i(k) = 0 for specified variable or for full vector
                variable_to_solve = solve_for_symbol if solve_for_symbol is not None else (
                    kx, ky, kz)
                sol = solveset(sp.Eq(lambda_i, 0),
                               variable_to_solve, domain=S.Reals)
                root_solutions.append((f"lambda_{i}=0", sol))
            except Exception as e:
                root_solutions.append(
                    (f"lambda_{i}=0", f"Error during solving: {e}"))

        return root_solutions

    # --- Fourier transformation to real space ---

    def compute_rspace_greens_symbolic_1d(self,
                                          z: Union[float, sp.Basic],
                                          z_prime: Union[float, sp.Basic],
                                          full_matrix: bool = False):
        """
        Compute the symbolic 1D real-space Green's function G(z, z′) via the residue theorem,
        assuming translational invariance in x and y. Only diagonal entries are returned in default mode. 
        If full matrix in the original basis is needed, enable full_matrix=True.

        Returns:
        - sp.Matrix: symbolic Green's function matrix in real space.
        """

        if not self.symbolic:
            warnings.warn(
                "Symbolic 1D G(z,z') computation is only supported in symbolic mode. Enable symbolic=True.")
            return []

        kvec = self.k_symbols
        kx, ky, kz = self.k_symbols

        q = self.q
        assert sp.simplify(z).is_real and sp.simplify(
            z_prime).is_real, "Both z and z′ must be real symbols or numbers"
        z_diff_sign = q  # default assumption: z-z'>0 for retarded GF and z<z' for advanced GF

        if not isinstance(z, sp.Symbol):
            assert not isinstance(
                z_prime, sp.Symbol), "Expected z' to be a real number, not a symbol."
            z = float(z)
            z_prime = float(z_prime)
            z_diff_sign = int(sp.sign(z - z_prime))
        else:
            assert isinstance(
                z_prime, sp.Symbol), "Expected z' to be instance of sp.Symbol."

        # solve_for = 2 means we solve for k_z = self.k_symbols[2]
        root_sets = self.compute_roots_greens_inverse(solve_for=2)
        poles_per_lambda = self._extract_valid_poles_from_root_solutions(
            root_sets)

        _, eigenvalues, _ = self.compute_eigen_greens_inverse(kvec)

        G_z_diag = [] # List for the diagonal entries of G(z,z'), each the solution of an integral
        has_contributions = False
        for i, (lambda_i, poles_i) in enumerate(zip(eigenvalues, poles_per_lambda)):
            lambda_i = sp.factor(lambda_i, extension=True)
            # ensure symbol consistency
            lambda_i = lambda_i.subs(dict(zip(self.k_symbols, [kx, ky, kz])))
            contrib, has_contributions = self._residue_sum_for_lambda(lambda_i, poles_i, z, z_prime, kz, z_diff_sign, has_contributions)

            if self.verbose:
                print(f"\nλ_{i}(k) = {lambda_i}")
                print(f"  Poles: {poles_i}")
                print(f"  Contribution to residue sum: {contrib}")

            G_z_diag.append(sp.I * contrib)

        if not has_contributions:
            warnings.warn("No poles passed the sign check; returning zero Green's function.")
            return sp.zeros(len(self.I))

        if all(val.equals(0) for val in G_z_diag):
            warnings.warn(
                "Green's function is identically zero: all residue contributions canceled.")

        G_z = sp.diag(*G_z_diag)
        # Note: Currently returning only diagonal Green's function G(z, z′)
        # Full matrix reconstruction from eigenbasis can be added if needed:
        if full_matrix:
            eigenbasis, _, _ = self.compute_eigen_greens_inverse(kvec)
            G_full = eigenbasis @ G_z @ invert_matrix(eigenbasis)
            return G_full

        return G_z
    
    def compute_rspace_greens_numeric_1D(self): #placeholder for now
        if self.symbolic:
            warnings.warn(
                "Numeric 1D G(z,z') computation is not supported in symbolic mode. Disable: symbolic=False.")
            return []

    # --- Internal utilities ---

    @staticmethod
    def _extract_valid_poles_from_root_solutions(root_solutions):
        """
        Extract valid poles from the output of compute_roots_greens_inverse.

        Returns:
            List[List[sp.Basic]]: One list per eigenvalue, containing its poles.
        """
        poles_per_lambda = []

        for label, solution in root_solutions:
            if isinstance(solution, sp.FiniteSet):
                poles = list(solution)
            elif hasattr(solution, 'is_EmptySet') and solution.is_EmptySet:
                poles = []
            else:
                # Could be a ConditionSet or error string; skip
                poles = []
            poles_per_lambda.append(poles)

        return poles_per_lambda
    
    def _residue_sum_for_lambda(self, lambda_i, poles_i, z, z_prime, kz_sym, z_diff_sign, has_contributions: bool):
        """
        Compute the residue sum for a single diagonal entry (lambda_i) of the Green's function.

        Parameters:
        - lambda_i: the i-th diagonal entry of G⁻¹(k)
        - poles_i: list of symbolic poles for that lambda_i
        - z, z_prime: real-space coordinates
        - kz_sym: the symbolic kz variable
        - z_diff_sign: sign(z - z') used for selecting poles
        - has_contributions: ensures there have previously been contributions

        Returns:
        - The total residue contribution for that diagonal element.
        - has_contributions: updated, if so, otherwise unchanged
        """
        contrib = 0
        for pole in poles_i:
            if z_diff_sign == sp.sign(sp.im(pole)):
                numerator = sp.exp(sp.I * pole * (z - z_prime))
                denom = sp.simplify(lambda_i / (kz_sym - pole))
                residue = numerator / denom.subs(kz_sym, pole)
                contrib += residue
                has_contributions = True
            elif self.verbose:
                print(f"Pole k_z = {pole} lies in the wrong half-plane and does not contribute.")
        return contrib, has_contributions

