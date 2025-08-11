import numpy as np
import sympy as sp
from sympy import solveset, S, pprint
from typing import Callable, Union, Optional
from utils import invert_matrix, print_symbolic_matrix, sanitize_vector, get_identity
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
        # validate identity
        if not (hasattr(self.I, "shape") and self.I.shape[0] == self.I.shape[1]):
            raise ValueError("Identity must be a square matrix.")
        self.N = int(self.I.shape[0]) # band size, e.g., 2 for spin-1/2 systems

        self.symbolic = symbolic
        self.omega = energy_level
        self.eta = infinitestimal
        self.q = 1 if retarded else -1

        # Choice of dimension determines default momentum symbols:
        self.d = dimension
        if not self.I == get_identity(size=self.d, symbolic=self.symbolic):
            raise ValueError("Input identity {self.I} does not have dimension {self.d}.")
        if self.d not in {1,2,3}:
            raise ValueError("Only 1D, 2D, and 3D systems are supported. Got dimension={self.d}.")
        if self.d == 1:
            self.k_symbols = [sp.symbols("k", real=True)]
        elif self.d in {2,3}:
            counter = self.d - 1
            self.k_symbols = sp.symbols(" ".join(f"k_{d}" for d in "xyz"[:counter]), real=True)

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
        # Ensure correct momentum dimensionality
        if len(momentum) != self.d:
            raise ValueError(f"Expected momentum vector of dimension {self.d}, got {len(momentum)}")

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
        if len(momentum) != self.d:
            raise ValueError(f"Expected momentum of dimension {self.d}, got {len(momentum)}")
        kvec = sanitize_vector(momentum, symbolic=self.symbolic)

        H_k = self.H(kvec)  # Hamiltonian at k
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
        - solve_for: Integer index (e.g., 0 for k₁), or None to solve for all momentum components simultaneously.

        Returns:
        - List of (label, solution) tuples, where each label is "lambda_i=0" for the i-th eigenvalue,
          and solution is either:
            * a symbolic solution set (FiniteSet or ConditionSet), or
            * an error message if solving fails.
        """
        if not self.symbolic:
            warnings.warn(
                "Root solving is only supported in symbolic mode. Enable symbolic=True.")
            return []
        assert len(self.k_symbols) == self.d

        if solve_for is not None:
            if not (0 <= solve_for < self.d):
                valid_indices = ", ".join(str(i) for i in range(self.d))
                raise ValueError(
                    f"'solve_for' must be one of {{{valid_indices}}} for a {self.d}D system. Got: {solve_for}"
                )
            variable_to_solve_for = self.k_symbols[solve_for]
        else:
            variable_to_solve_for = None

        # Define symbolic momentum components 
        k = sp.Matrix(self.k_symbols) # e.g., Matrix([k_x, k_y, k_z]) or fewer

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
            if lambda_i.free_symbols.isdisjoint(set(self.k_symbols)):
                # Skip solving or return empty solution set
                root_solutions.append((f"lambda_{i}=0", sp.FiniteSet()))
                continue

            if not lambda_i.is_polynomial(variable_to_solve_for):
                warnings.warn(
                    f"Solving λ_{i}(k) = 0 may fail: expression is not polynomial in {variable_to_solve_for}.", stacklevel=2)

            try:
                # Solve λ_i(k) = 0 for specified variable or for full vector
                variable = variable_to_solve_for if variable_to_solve_for is not None else tuple(self.k_symbols)
                sol = solveset(sp.Eq(lambda_i, 0),
                               variable, domain=S.Reals)
                root_solutions.append((f"lambda_{i}=0", sol))
            except Exception as e:
                root_solutions.append(
                    (f"lambda_{i}=0", f"Error during solving: {e}"))

        return root_solutions

    # --- Fourier transformation to real space ---

    def compute_rspace_greens_symbolic_1d_along_last_dim(self,
                                                         z: Union[float, sp.Basic],
                                                         z_prime: Union[float, sp.Basic],
                                                         full_matrix: bool = False):
        """
        Compute the symbolic 1D real-space Green's function G(z, z′) via the residue theorem.

        Assumes translational invariance in all but the last spatial dimension, performing a 1D Fourier 
        transform along k_{d-1}. Supports 1D, 2D, and 3D systems.
        
        Only diagonal entries are returned in default mode. 
        If full matrix in the original basis is needed, enable full_matrix=True.

        Parameters:
        - z, z′: Real or real symbolic coordinates along the last spatial dimension.
        - full_matrix: If True, reconstruct the full Green's function matrix (not just the diagonal).

        Returns:
        - G(z, z′): The symbolic real-space Green's function matrix.
        """

        if not self.symbolic:
            warnings.warn(
                "Symbolic 1D G(z,z') computation is only supported in symbolic mode. Enable symbolic=True.")
            return []
        assert self.d >= 1, "Cannot perform real-space transform in zero-dimensional system."

        kvec = sp.Matrix(self.k_symbols)
        k_dir = self.k_symbols[self.d - 1]  # direction of real-space transform (last component)
        
        if self.verbose:
            print(f"\nPerforming 1D Fourier transform in {self.d} dimensions over variable {k_dir}.")

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
        root_sets = self.compute_roots_greens_inverse(solve_for=self.d - 1)
        poles_per_lambda = self._extract_valid_poles_from_root_solutions(
            root_sets)

        _, eigenvalues, _ = self.compute_eigen_greens_inverse(kvec)

        G_z_diag = [] # List for the diagonal entries of G(z,z'), each the solution of an integral
        has_contributions = False
        for i, (lambda_i, poles_i) in enumerate(zip(eigenvalues, poles_per_lambda)):
            lambda_i = sp.factor(lambda_i, extension=True)
            contrib, has_contributions = self._residue_sum_for_lambda(lambda_i, poles_i, z, z_prime, k_dir, z_diff_sign, has_contributions)

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
        Apply the residue theorem to compute the contribution to G(z, z′) from one eigenvalue λᵢ.
        This method of calculating the residue is based on the assumption that the diagonal
        entries λᵢ(k) of G⁻¹(k) are polynomials in the integration variable (e.g. λᵢ(k_z)= (k_z)^2/(2m)).

        Parameters:
        - lambda_i: Diagonal entry λᵢ(k) of G⁻¹(k). Must be polynomial in the integration variable!
        - poles_i: Valid poles of λᵢ
        - z, z′: Coordinates in real space (must be real-valued or symbolic real)
        - kz_sym: Momentum variable to integrate over (e.g., k_z)
        - z_diff_sign: Determines correct half-plane for the contour
        - has_contributions: Tracks whether any pole has contributed

        Returns:
        - contrib: Total residue contribution to G_{ii}(z, z′)
        - has_contributions: Updated flag
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
                print(f"Pole {kz_sym} = {pole} lies in the wrong half-plane and does not contribute.")
        return contrib, has_contributions

