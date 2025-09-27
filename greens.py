import numpy as np
import sympy as sp
from sympy import pprint
from typing import Callable, Union
from utils import invert_matrix, print_symbolic_matrix, sanitize_vector
import warnings
# reconstruction tolerance for eigen-decomp checks
NUM_EIG_TOL = 1e-8
INFINITESIMAL = 1e-6  # default infinitesimal if none provided


class GreensFunctionCalculator:
    def __init__(self,
                 hamiltonian: Callable[[Union[list, np.ndarray, sp.Matrix]], Union[np.ndarray, sp.Matrix]],
                 identity: Union[np.ndarray, sp.Matrix],
                 symbolic: bool,
                 # omega
                 energy_level: Union[float, sp.Basic],
                 # eta
                 broadening: Union[float, sp.Basic] = None,
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
        - identity: identity matrix (NxN) for the appropriate backend, where N is the band size
        - symbolic: whether to use symbolic (sympy) or numeric (numpy) mode
        - energy_level: scalar ω
        - infinitestimal: small η > 0 to define the imaginary part
        - retarded: if True computes retarded Green's function; else advanced
        - dimension: spatial dimension of the system (1, 2, or 3), defaults to 3
        - verbose: if True, prints intermediate matrix states for debugging
        """
        self.H = hamiltonian
        self.I = identity
        # validate identity
        if not (hasattr(self.I, "shape") and self.I.shape[0] == self.I.shape[1]):
            raise ValueError(f"Identity must be a square matrix.")
        # band size, e.g., 2 for spin-1/2 systems
        self.N = int(self.I.shape[0])

        self.symbolic = symbolic
        # Ensure identity is in the correct format
        if self.symbolic and isinstance(self.I, np.ndarray):
            self.I = sp.Matrix(self.I)
        elif not self.symbolic and isinstance(self.I, sp.MatrixBase):
            self.I = np.asarray(np.array(self.I.tolist(), dtype=complex))

        self.omega = energy_level
        if broadening is not None:
            self.eta = broadening
            if self.eta < 0:
                raise ValueError("Broadening η must not be negative.")
            elif broadening == 0:
                warnings.warn(
                    "Broadening η is zero; Green's function may be ill-defined at poles.")

        elif self.symbolic:
            self.eta = sp.symbols("eta", positive=True)
            warnings.warn("No broadening η provided; using symbolic η > 0.")
        else:
            self.eta = INFINITESIMAL
            warnings.warn(
                f"No broadening η provided; defaulting to η={self.eta}.")
        self.q = 1 if retarded else -1

        # Choice of dimension determines default momentum symbols:
        self.d = int(dimension)
        if self.d not in (1, 2, 3):
            raise ValueError(
                f"Only 1D, 2D, and 3D systems are supported. Got dimension={self.d}.")

        # Canonical momentum symbols used internally for solving:
        # k_symbols[0] = "k_x", k_symbols[1] = "k_y", k_symbols[2] = "k_z"
        # Only relevant for symbolic mode.
        if self.symbolic:
            names = ["k"] if self.d == 1 else [
                f"k_{ax}" for ax in "xyz"[:self.d]]
            # not limited to real numbers since complex values must be allowed for integration
            self.k_symbols = sp.symbols(" ".join(names))
            # For consistency in code paths, make it indexable like a list
            if isinstance(self.k_symbols, sp.Symbol):
                self.k_symbols = [self.k_symbols]
            assert isinstance(self.k_symbols, (list, tuple)
                              ) and len(self.k_symbols) == self.d
        else:
            # numeric path: you still need the length for checks
            self.k_symbols = [None] * self.d

        # for easy debugging along the way
        self.verbose = verbose

        self.green_type = "retarded (+iη)" if self.q == 1 else "advanced (−iη)"

    def info(self):
        """
        Print a summary of the internal configuration of this calculator instance.
        """
        print("\nGreensFunctionCalculator configuration:")
        print("======================================")
        mode = "Symbolic" if self.symbolic else "Numeric"
        print(f"Computation mode           : {mode}")
        print(f"Verbose mode               : {self.verbose}")
        print(f"Energy ω                   : {self.omega}")
        print(f"Infinitesimal broadening η : {self.eta}")
        print(f"Green's function type      : {self.green_type}")
        print(f"Dimension d                : {self.d}")
        print(f"Band size N                : {self.N}")
        print(
            f"H(k) callable              : {'Yes' if callable(self.H) else 'No'}")
        print(f"Momentum symbols           : {self.k_symbols}")
        print("======================================\n")

    # --- GF computation in k-space ---

    def compute_kspace_greens_function(self, momentum: Union[np.ndarray, sp.Matrix] = None) -> Union[np.ndarray, sp.Matrix]:
        """
        Compute the Green's function for a single-particle Hamiltonian in momentum space by inverting
        (omega + q*i*eta - H(k)), where q = ±1 for retarded/advanced GF.

        Parameters
        ----------
        momentum: np.ndarray or sp.Matrix
            value at which the Hamiltonian is evaluated
            If None, defaults to k symbols in symbolic mode and raises a ValueError in numeric mode.


        Returns
        ---------
        G(k) as np.ndarray or sp.Matrix
            Green's function in momentum space

        Raises
        ------
        ValueError
            If called in numeric mode without a specific momentum value.
        """
        if momentum is None:
            if self.symbolic:
                momentum = sp.Matrix(self.k_symbols)
            else:
                raise ValueError(
                    "Momentum must be provided in numeric mode (symbolic=False).")
        if self.d != 1:
            # Sanitize momentum vector input for correct format
            momentum = sanitize_vector(momentum, self.symbolic)
            # Ensure correct momentum dimensionality
            if len(momentum) != self.d:
                raise ValueError(
                    f"Expected momentum vector of dimension {self.d}, got {len(momentum)}")

        H_k = self.H(momentum)  # Hamiltonian at momentum k
        # ensure indexable even in single-channel case
        H_k = [H_k] if self.N == 1 else H_k

        # convert to backend-specific matrix/array
        H_k = sp.Matrix(H_k) if self.symbolic else np.asarray(
            H_k, dtype=complex)
        if H_k.shape != (self.N, self.N):
            raise ValueError(
                f"H(k) must be {self.N}x{self.N}, got {H_k.shape}.")

        imaginary_unit = sp.I if self.symbolic else 1j
        G_inv = (self.omega + self.q * self.eta *
                 imaginary_unit) * self.I - H_k
        G_k = invert_matrix(G_inv, symbolic=self.symbolic)

        if self.verbose:
            print("\nComputing Green's function at momentum k")
            print("\nwith k = ", momentum)
            print_symbolic_matrix(
                H_k, name="H(k)") if self.symbolic else print("H(k) =\n", H_k)
            if self.q == 1:
                print_symbolic_matrix(
                    G_inv, name="( ω + iη - H(k) )") if self.symbolic else print("( ω ± iη - H(k) ) =\n", G_inv)
            else:
                print_symbolic_matrix(
                    G_inv, name="( ω - iη - H(k) )") if self.symbolic else print("( ω ± iη - H(k) ) =\n", G_inv)

        return G_k

    def compute_eigen_greens_inverse(self, momentum: Union[np.ndarray, sp.Matrix] = None) -> Union[np.ndarray, sp.Matrix, list]:
        """
        Diagonalize the inverse Green's function matrix to obtain its eigenbasis and eigenvalues.
        Useful for identifying poles and simplifying root solving.

        Parameters
        ----------
        momentum: np.ndarray or sp.Matrix
            value at which the Hamiltonian is evaluated
            If None, defaults to k symbols in symbolic mode and raises a ValueError in numeric mode.


        Returns
        ---------
        - eigenbasis (matrix of eigenvectors) of G⁻¹(k)
        - eigenvalues of G⁻¹(k)
        - G⁻¹(k) diagonalized

        Raises
        ------
        ValueError
            If called in numeric mode without a specific momentum value.
        """
        if momentum is None:
            if self.symbolic:
                momentum = sp.Matrix(self.k_symbols)
            else:
                raise ValueError(
                    "Momentum must be provided in numeric mode (symbolic=False).")
        # 1) momentum validation
        if self.d != 1:
            # Sanitize momentum input
            momentum = sanitize_vector(momentum, self.symbolic)
            # Ensure correct momentum dimensionality
            if len(momentum) != self.d:
                raise ValueError(
                    f"Expected momentum vector of dimension {self.d}, got {len(momentum)}")

        # 2) H(k) build + shape check
        H_k = self.H(momentum)
        H_k = [H_k] if self.N == 1 else H_k  # ensure indexable for 1D

        if self.symbolic:
            H_k = sp.Matrix(H_k)
        else:
            H_k = np.asarray(H_k, dtype=complex)

        if H_k.shape != (self.N, self.N):
            raise ValueError(
                f"H(k) must be {self.N}x{self.N}, got {H_k.shape}.")

        # 3) G^{-1}(k)
        if self.symbolic:
            G_inv = (self.omega + self.q * self.eta * sp.I) * self.I - H_k
        else:
            G_inv = (self.omega + self.q * self.eta * 1j) * self.I - H_k

        # 4) Eigendecomposition
        # Symbolic mode
        if self.symbolic:
            evects = G_inv.eigenvects()
            pairs = []
            for lam, mult, vecs in evects:
                for v in vecs:
                    v = sp.Matrix(v)
                    if v.norm() != 0:
                        v = v / v.norm()
                    else:
                        raise ValueError("Zero-norm eigenvector encountered.")
                    pairs.append((sp.simplify(lam), v))

            if len(pairs) != self.N:
                raise ValueError(
                    "G^{-1}(k) is not diagonalizable: insufficient eigenvectors.")

            # Sorting for a reproducible order
            pairs.sort(key=lambda t: sp.default_sort_key(t[0]))

            eigenvalues = [lam for lam, _ in pairs]
            # create eigenbasis matrix from stacking eigenvectors
            P = sp.Matrix.hstack(*[v for _, v in pairs])

            try:
                P_inv = P.inv()
            except Exception:
                raise ValueError(
                    "G^{-1}(k) is not diagonalizable: eigenbasis is singular, i.e. not invertible.")

            D = sp.diag(*eigenvalues)

            #####################################################
            # Consistency check:
            # D (from eigenvalues) vs D_recon = (P^{-1} G^{-1} P)
            #####################################################
            D_recon = sp.simplify(P_inv * G_inv * P)
            # (1) Off-diagonals must vanish exactly (or simplify to zero)
            offdiag = D_recon - sp.diag(*[D_recon[i, i]
                                        for i in range(self.N)])
            if not offdiag.equals(sp.zeros(self.N)):
                if not sp.simplify(offdiag).is_zero_matrix:
                    raise ValueError(
                        "Eigendecomposition inconsistent: off-diagonal terms remain in P^{-1} G^{-1} P.")
            # (2) Diagonals must match eigenvalues (after simplification)
            for i in range(self.N):
                diff = sp.simplify(D_recon[i, i] - D[i, i])
                # equals(0) can be None; also try is_zero/simplify to be safe
                if not (diff.equals(0) or diff.is_zero):
                    raise ValueError(
                        "Eigendecomposition inconsistent: diagonal of P^{-1} G^{-1} P does not match eigenvalues.")
            # (3) Final reconstruction check
            residual = (P * D * P_inv - G_inv)
            if not residual.equals(sp.zeros(self.N)):
                if not sp.simplify(residual).is_zero_matrix:
                    raise ValueError(
                        "Eigendecomposition failed: P*D*P^{-1} != G^{-1}(k).")

            if self.verbose:
                print("\nDiagonal elements (eigenvalues) of G⁻¹(k):")
                for i, lam in enumerate(eigenvalues):
                    print(f"λ[{i}] = {sp.simplify(lam)}")

            return P, eigenvalues, D

        # Numeric mode
        else:
            vals, vecs = np.linalg.eig(G_inv)
            # Sorting by real part first, imaginary part second gives a fixed, reproducible order
            idx = np.lexsort((vals.imag, vals.real))
            vals = vals[idx]
            vecs = vecs[:, idx]

            P = np.array(vecs, dtype=complex)

            # Ill-conditioned warning:
            condP = np.linalg.cond(P)  # condition number of eigenbasis
            if not np.isfinite(condP) or condP > 1e12:
                warnings.warn(
                    f"Ill-conditioned eigenbasis (cond={condP:.2e}). Results may be unstable.")

            P_inv = invert_matrix(P, symbolic=False)
            D = np.diag(vals.astype(complex))

            #####################################################
            # Consistency check (up to numerical error):
            # D (from eigenvalues) vs D_recon = (P^{-1} G^{-1} P)
            #####################################################
            D_recon = P_inv @ G_inv @ P
            # (1) Off-diagonal energy should be ~0
            offdiag = D_recon.copy()
            np.fill_diagonal(offdiag, 0.0)
            offdiag_norm = np.linalg.norm(offdiag)
            if offdiag_norm > NUM_EIG_TOL:
                raise ValueError(
                    f"Eigendecomposition inconsistent: off-diagonal norm {offdiag_norm:.2e} exceeds {NUM_EIG_TOL:.1e}"
                )
            # (2) Diagonal entries should match eigenvalues within tolerance (order already enforced by sorting)
            diag_diff = np.diag(D_recon) - np.diag(D)
            # normalization for appropriate tolerance scaling:
            rel_err = np.linalg.norm(diag_diff) / \
                max(1.0, np.linalg.norm(np.diag(D)))
            if rel_err > NUM_EIG_TOL:
                raise ValueError(
                    f"Eigendecomposition inconsistent: diagonal mismatch rel. error {rel_err:.2e} exceeds {NUM_EIG_TOL:.1e}"
                )
            # (3) Final reconstruction check
            # Frobenius norm ought to vanish
            residual_norm = np.linalg.norm(P @ D @ P_inv - G_inv)
            # normalization for appropriate tolerance scaling
            den = max(1.0, np.linalg.norm(G_inv))
            if residual_norm / den > NUM_EIG_TOL:
                raise ValueError(
                    "Eigendecomposition failed: reconstruction error above tolerance.")

            if self.verbose:
                print("\nDiagonal elements (eigenvalues) of G⁻¹(k):")
                for i, lam in enumerate(vals):
                    print(f"λ[{i}] = {lam}")

            return P, vals, D

    def compute_roots_greens_inverse(self, solve_for: int = None, case_assumptions: list = None) -> list[tuple[str, sp.Set]]:
        """
        Solve for the roots of the eigenvalues of G^{-1}(k) with respect to ONE momentum component,
        i.e., values of momentum where one or more eigenvalues of the inverse Green's function vanish.
        Those poles correspond to the dispersion relations defining the band structure of the material.
        Only single-variable solving is supported!
        Numeric mode is not supported!
        May return a ConditionSet instead of a FiniteSet if the eigenvalues are not a polynomial 
        in the chosen variable.

        Parameters
        ----------
        solve_for : int;
            Index of the momentum component to solve for: 0..(d-1)
            If None, defaults to the last dimension (d-1).

        Returns
        -------
        List[Tuple[str, sp.Set]]
            A list of (label, solution_set) pairs, one per eigenvalue λ_i, where the set contains the
            k_{solve_for} in the complex domain that solve λ_i = 0. 
            Non-polynomial cases may return a ConditionSet.

        Raises
        ------
        TypeError
            If called in numeric mode (symbolic=False) or if `solve_for` is not an int.
        ValueError
            If `solve_for` is out of range [0, d-1].
        """
        if not self.symbolic:
            warnings.warn(
                "Root solving is only supported in symbolic mode. Enable symbolic=True.")
            return []  # no error raised but empty list returned

        if solve_for is None:
            solve_for = self.d - 1  # default to last dimension
        # validate index
        if not isinstance(solve_for, int):
            raise TypeError(
                f"'solve_for' must be an int in [0, {self.d-1}] (k-dimension index).")
        if solve_for < 0 or solve_for >= self.d:
            valid_indices = ", ".join(str(i) for i in range(self.d))
            raise ValueError(
                f"'solve_for' out of range: got {solve_for}, valid indices are {{{valid_indices}}}.")

        k_var = self.k_symbols[solve_for]  # variable to solve for

        if self.verbose:
            print(
                f"\nSolving for roots of G⁻¹(k) eigenvalues with respect to k_symbols[{solve_for}] = {k_var}...")

        # Define symbolic momentum components
        k = sp.Matrix(self.k_symbols)  # e.g., Matrix([k_x, k_y, k_z]) or fewer

        # Compute eigenvalues of the inverse Green's function
        _, eigenvalues, _ = self.compute_eigen_greens_inverse(k)

        root_solutions = []
        non_empty_flag = 0  # count how many eigenvalues have non-empty solution sets
        for i, lambda_i in enumerate(eigenvalues):
            # simplify for readability and solving
            lambda_i = sp.simplify(lambda_i)
            if self.verbose:
                print(f"\nThe free symbols in eigenvalue λ[{i}] are: {lambda_i.free_symbols}.")
            # a) Short circuit if λ_i has no dependence on k_var
            if k_var not in lambda_i.free_symbols:
                if self.verbose:
                    print(f"This set does not include the variable we solve for: {k_var}.")
                # check for all-together vanishing eigenvalue
                if lambda_i.equals(0) or lambda_i.is_zero:
                    raise ValueError(
                        f"Eigenvalue λ_{i} is identically zero for all {k_var}, indicating a singular G⁻¹(k).")
                # λ_i is constant in the variable to solve for
                warnings.warn(
                    f"The {i}th eigenvalue lambda_{i}={lambda_i} has no dependence {k_var}; returning empty solution set.")
                # Return empty solution set
                root_solutions.append((f"lambda_{i}=0", sp.FiniteSet()))
                continue
            non_empty_flag = 1

            # b) Warn if λ_i is not polynomial in k_var
            if not lambda_i.is_polynomial(k_var):
                warnings.warn(
                    f"Solving λ_{i}(k) = 0 may fail: expression is not polynomial in {k_var}.", stacklevel=2)

            # c) Solve for roots of lambda_i = 0
            try:
                lam_simpl = sp.simplify(lambda_i)
                try:
                    # polynomial attempt
                    # let SymPy pick the domain
                    poly = sp.Poly(lam_simpl, k_var)
                    if poly.total_degree() > 0:
                        # dict {root: multiplicity}
                        roots_dict = sp.roots(poly.as_expr(), k_var)
                        # expand multiplicities into a list, to match your previous FiniteSet(*roots)
                        roots_list = []
                        for r, m in roots_dict.items():
                            roots_list.extend([sp.simplify(r)] * int(m))
                        solset = sp.FiniteSet(*roots_list)
                    else:
                        solset = sp.S.Complexes if sp.simplify(
                            lam_simpl) == 0 else sp.EmptySet
                except sp.PolynomialError:
                    # general solve
                    solset = sp.solveset(
                        sp.Eq(lam_simpl, 0), k_var, domain=sp.S.Complexes)
                    if isinstance(solset, sp.ConditionSet):
                        warnings.warn(
                            f"λ[{i}] is not polynomial in {k_var}; returning a ConditionSet.",
                            stacklevel=2
                        )
                root_solutions.append((f"lambda_{i}=0", solset))
            except Exception as e:
                # fallback if something really unexpected happens
                root_solutions.append(
                    (f"lambda_{i}=0", f"Error during solving: {e}"))

        if non_empty_flag == 0:
            warnings.warn(
                "None of the eigenvalues depend on the variable to solve for; all solution sets are empty.")
        if self.verbose:
            print("\nRoots of the Hamiltonian:")
            pprint(root_solutions, use_unicode=True)

        return root_solutions

    # --- Fourier transformation to real space ---
    # -- Symbolic 1D real-space transform --
    def compute_rspace_greens_symbolic_1d_along_last_dim(self,
                                                         z: Union[float, sp.Basic],
                                                         z_prime: Union[float, sp.Basic],
                                                         z_diff_sign: int = None,
                                                         full_matrix: bool = False,
                                                         case_assumptions: list = None):
        """
        Compute the symbolic 1D real-space Green's function G(z, z′) via the residue theorem.

        Assumes translational invariance in all but the last spatial dimension, performing a 1D Fourier 
        transform along k_{d-1}. Supports 1D, 2D, and 3D systems.

        Only diagonal entries are returned in default mode. 
        If full matrix in the original basis is needed, enable full_matrix=True.

        Parameters
        ----------
        z, z′: Real numbers or real symbols; 
            coordinates along the last spatial dimension.
        z_diff_sign: int;
            Sign of (z-z′) to determine contour closure direction:
            If None provided, it defaults to q.
            Usually, q=+1, i.e. z > z′, since GF calculator defaults to retarded (+iη).
        full_matrix: boolean;
            If True, reconstruct the full Green's function matrix in its original basis (not just the diagonal form).

        Returns:
        ----------
        G(z, z′): matrix;
            The symbolic real-space Green's function matrix.
        """
        # optional: a list of SymPy assumptions to resolve ambiguous points for the solver
        if case_assumptions is None:
            case_assumptions = []

        if not self.symbolic:
            warnings.warn(
                "Symbolic 1D G(z,z') computation is only supported in symbolic mode. Enable symbolic=True.")
            return []
        assert self.d >= 1, "Cannot perform real-space transform in zero-dimensional system."

        if z_diff_sign is None:
            z_diff_sign = self.q  # default to q
            warnings.warn(
                f"No z_diff_sign provided; defaulting to the assumption z_diff_sign={self.q}, since the GF is {self.green_type}.")

        kvec = sp.Matrix(self.k_symbols)
        # direction of real-space transform (last component)
        k_dir = self.k_symbols[self.d - 1]

        if self.verbose:
            print(
                f"\nPerforming 1D Fourier transform of the {self.green_type} Green's function over variable {k_dir}.")

        z_sym, zp_sym = sp.sympify(z), sp.sympify(z_prime)
        assert z_sym.is_real is not False and zp_sym.is_real is not False, "Both z and z′ must be real symbols or numbers"

        if not isinstance(z, sp.Symbol):
            assert not isinstance(
                z_prime, sp.Symbol), "Expected z' to be a real number, not a symbol."
            z = float(z)
            z_prime = float(z_prime)
            z_diff = z - z_prime
            sig = int(sp.sign(z_diff))
            if sig == 0:
                warnings.warn("z and z' are equal; results may be singular.")
            assert z_diff_sign == sig, f"Expected the sign of (z-z')={z-z_prime} to match z_diff_sign={z_diff_sign}. Adjust z_diff_sign to match sign(z-z')."

        else:
            assert isinstance(
                z_prime, sp.Symbol), "Expected z' to be instance of sp.Symbol since z is one."

        if self.verbose:
            print(
                "( ω + iη - H(k) ) will be diagonalized to evaluate residues for the Fourier integral.") if self.green_type == "retarded (+iη)" else print("( ω - iη - H(k) ) will be diagonalized to evaluate residues for the Fourier integral.")

        _, eigenvalues, _ = self.compute_eigen_greens_inverse(kvec)

        if self.verbose:
            if z_diff_sign == 1:
                print(f"\nThe Fourier integral over {k_dir} is computed using contour integration,") 
                print("utilizing Jordan's Lemma and the Residue Theorem.")
                print("The contour is closed in the upper half-plane, since z>z'.")
            elif z_diff_sign == -1:
                print(f"\nThe Fourier integral over {k_dir} is computed using contour integration,") 
                print("utilizing Jordan's Lemma and the Residue Theorem.")
                print("The contour is closed in the lower half-plane, since z<z'.")
            else:
                warnings.warn(
                    "The sign of z-z' cannot be determined. More information is needed for contour integration")

        # List for the diagonal entries of G(z,z'), each the solution of an integral
        G_z_diag = []
        has_contributions = False

        for i, lambda_i in enumerate(eigenvalues):
            contrib, contributed_any = self._residue_sum_for_lambda(
                lambda_i, z, z_prime, k_dir, z_diff_sign, case_assumptions=case_assumptions)
            has_contributions = has_contributions or contributed_any
            assert contrib == 0 if not contributed_any else True, "If no poles contributed, the contribution must be zero."
            G_z_diag.append(contrib)

            if self.verbose:
                print(f"\nThe eigenvalue λ_{i+1}(k) = {lambda_i}")
                print(f"contributes to the residue sum with: {contrib}") if contrib != 0 else print("does not contribute to the residue sum.")

        if not has_contributions:
            warnings.warn(
                "No poles passed the sign check; returning zero Green's function.")
            assert sp.diag(*G_z_diag) == sp.zeros(self.N), "Expected zeros on the diagonal if no poles contributed."

        elif all((val.is_zero is True) for val in G_z_diag):
            warnings.warn(
                "Green's function is identically zero: all residue contributions canceled out.")
            assert has_contributions == True, "Expected has_contributions=True if some poles contributed, otherwise earlier warning should have been triggered."

        G_z = sp.diag(*G_z_diag)

        if self.verbose:
            print(f"\nThe real space {self.green_type} Green's function in the last spatial dimension is:")
            print(f"G({z}, {z_prime}) = {G_z}")
            print(f"where it is assumed that z {'>' if z_diff_sign==1 else '<'} z'.")
            print("If different assumptions on z and z' are needed, rerun with the appropriate z_diff_sign parameter.")
            print(f"with additional assumptions: {case_assumptions}") if case_assumptions else print("with no additional case assumptions fed to the solver.")
            print("If assumptions need to be altered, rerun with the appropriate case_assumptions parameter.")
            if not full_matrix:
                print("\nNote: Only diagonal entries are returned by default.")  

        # Note: Currently returning only diagonal Green's function G(z, z′)
        # Full matrix reconstruction from eigenbasis can be added if needed:
        if full_matrix:
            eigenbasis, _, _ = self.compute_eigen_greens_inverse(kvec)
            G_full = eigenbasis @ G_z @ invert_matrix(
                eigenbasis, symbolic=True)
            if self.verbose:
                print("\nThe full Green's function matrix in the original basis is:")
                print(f"G({z}, {z_prime}) = {G_full}")
            return G_full

        return G_z

    def compute_rspace_greens_symbolic_1d(self, z, z_prime, z_diff_sign=None, full_matrix: bool = False):
        """
        Wrapper around compute_rspace_greens_symbolic_1d_along_last_dim that
        returns results in the legacy format expected by tests:
        a list of (label, expression) tuples.

        Parameters
        ----------
        z : float or sympy.Symbol
            Position along the chosen 1D direction.
        z_prime : float or sympy.Symbol
            Reference position along the same direction.
        full_matrix : bool, optional
            If True, return the full Green's function matrix. If False, return
            only the diagonal entries. Default is False.

        Returns
        -------
        list of (str, sympy.Expr)
            Tuples labeling each matrix element (e.g., "G_00") with its
            corresponding expression.
        """
        G = self.compute_rspace_greens_symbolic_1d_along_last_dim(
            z, z_prime, z_diff_sign, full_matrix=full_matrix)

        # If numeric mode: nothing implemented, return []
        if isinstance(G, list):
            return []

        results = []
        if full_matrix:
            for i in range(G.shape[0]):
                for j in range(G.shape[1]):
                    results.append((f"G_{i}{j}", G[i, j]))
        else:
            for i in range(G.shape[0]):
                results.append((f"G_{i}{i}", G[i, i]))

        return results

    # -- Numeric 1D real-space transform --
    def compute_rspace_greens_numeric_1D(self,
                                         z: float,
                                         z_prime: float,
                                         full_matrix: bool = False):
        '''
        placeholder function for later implementation
        '''
        if self.symbolic:
            warnings.warn(
                "Numeric 1D G(z,z') computation is not supported in symbolic mode. Disable: symbolic=False.")

        warnings.warn("Numeric 1D G(z,z') not implemented yet; returning [].")
        return []

    # --- Internal utilities ---

    def _residue_sum_for_lambda(self, lambda_i, z, z_prime, kz_sym, z_diff_sign, case_assumptions: list = None):
        """
        Apply the residue theorem to compute the contribution to G(z, z′) from one eigenvalue λᵢ.
        This method of calculating the residue is based on the assumption that the diagonal
        entries λᵢ(k) of G⁻¹(k) are polynomials in the integration variable "kz_sym" (e.g. λᵢ(k_z)= (k_z)^2/(2m)).
        Therefore, G(k) has poles at the roots of λᵢ(k). The multiplicity of those poles are taken into account.
        Only poles in the correct half-plane (sign(im(k0)) == z_diff_sign) contribute.

        Parameters:
        - lambda_i: Diagonal entry λᵢ(k) of G⁻¹(k). Must be polynomial in the integration variable!
        - poles_i: Valid poles of λᵢ
        - z, z′: Coordinates in real space (must be real-valued or symbolic real)
        - kz_sym: Momentum variable to integrate over (e.g., k_z)
        - z_diff_sign: Determines correct half-plane for the contour
        - has_contributions: Tracks whether any pole has contributed

        Returns:
        - contrib: Total residue contribution to G_{ii}(z, z′), that is the residue sum multiplied by i.
        - has_contributions: Updated flag
        """
        contributed_any = False

        #if case_assumptions is None:
        #    case_assumptions = []

        ## store (k0, m, res_expr) to resolve ambiguity:
        #ambiguous = []

        # Short-circuit if λ_i has no dependence on kz_sym:
        if kz_sym not in lambda_i.free_symbols:
                if self.verbose:
                    print(f"The eigenvalue {lambda_i} does not depend on the integration variable: {kz_sym}.")
                contrib = 0
                return contrib, contributed_any

        z_diff = z - z_prime
        phase = sp.exp(sp.I * kz_sym * z_diff)

        if not lambda_i.is_polynomial(kz_sym):
            # Not polynomial (SymPy can’t reliably find poles)
            # Triggers unevaluated integral fallback
            warnings.warn(
                f"Eigenvalue is not polynomial in k; returning unevaluated Fourier integral.")
            expr = sp.Integral(phase / sp.simplify(lambda_i),
                               (kz_sym, -sp.oo, sp.oo)) / (2*sp.pi)
            contributed_any = True
            if self.verbose:
                print(
                    f"\nUnevaluated integral expression for G(k) diagonal entry: {expr}")
                print("The Residue Theorem was not applied.")
            return expr, contributed_any

        # dict {root: multiplicity}
        roots_with_mult = sp.roots(sp.simplify(lambda_i), kz_sym)
        if self.verbose:
            print(
                f"\nFound roots of eigenvalue {lambda_i}: {roots_with_mult} (root: multiplicity)")

        residue_sum = 0
        for k0, m in roots_with_mult.items():  # roots k0 with their multiplicity m
            # Residue formula for pole of order m:
            # Res = 1/(m-1)! * d^{m-1}/dk^{m-1} [ (k-k0)^m * phi / lambda_i(k) ] at k=k0
            expr = sp.simplify(((kz_sym - k0)**m) * phase / lambda_i)
            if m == 1:
                res = sp.simplify(expr.subs(kz_sym, k0))  # zero-th derivative
            else:
                deriv = sp.diff(expr, (kz_sym, m - 1))
                res = sp.simplify(deriv.subs(kz_sym, k0) / sp.factorial(m - 1))
            # Half-plane selector
            sgn = sp.sign(sp.im(k0))
            if sgn in (sp.Integer(1), sp.Integer(-1)):
                if int(sgn) == z_diff_sign:
                    residue_sum += res
                    contributed_any = True
                elif self.verbose:
                    print(
                        f"\nPole k={k0} (m={m}) lies in wrong half-plane; skipped.")
            else:
                #ambiguous.append((k0, m, res))
                raise ValueError(
                    "Indeterminate pole selection: sign(Im(k0)) is unknown. "
                    "Provide further assumptions (e.g. sp.Q.positive(omega - V_F)) to resolve."
                )

            # If using assumptions, resolve ambiguous ones now
            #if ambiguous:
            #    if not case_assumptions:
            #        raise ValueError(
            #            "Indeterminate pole selection and no case_assumptions provided. "
            #            "Pass e.g. case_assumptions=[sp.Q.positive(omega - V_F)]"
            #        )
            #    extra = 0
            #    for a in case_assumptions:
            #        with sp.assuming(a):
            #            for (k0, m, res_expr) in ambiguous:
            #                s = sp.sign(sp.im(k0))
            #                if s in (sp.Integer(1), sp.Integer(-1)) and int(s) == z_diff_sign:
            #                    extra += sp.simplify(res_expr)
            #                    contributed_any = True
            #    residue_sum += extra


        contrib = sp.I * residue_sum  # factor of i from residue theorem

        return contrib, contributed_any
