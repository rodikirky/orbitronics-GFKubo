import numpy as np
import sympy as sp
from sympy import solveset, S, pprint
from typing import Callable, Union, Optional
from utils import invert_matrix, print_symbolic_matrix, sanitize_vector, get_identity
import warnings
NUM_EIG_TOL = 1e-8  # reconstruction tolerance for eigen-decomp checks


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
        self.N = int(self.I.shape[0]) # band size, e.g., 2 for spin-1/2 systems

        self.symbolic = symbolic
        # Ensure identity is in the correct format
        if self.symbolic and isinstance(self.I, np.ndarray):
            self.I = sp.Matrix(self.I)
        elif not self.symbolic and isinstance(self.I, sp.MatrixBase):
            self.I = np.asarray(np.array(self.I.tolist(), dtype=complex))

        self.omega = energy_level
        self.eta = infinitestimal
        self.q = 1 if retarded else -1

        # Choice of dimension determines default momentum symbols:
        self.d = int(dimension)
        if self.d not in (1, 2, 3):
            raise ValueError(f"Only 1D, 2D, and 3D systems are supported. Got dimension={self.d}.")
        if self.symbolic:
            names = ["k"] if self.d == 1 else [f"k_{ax}" for ax in "xyz"[:self.d]]
            self.k_symbols = sp.symbols(" ".join(names), real=True)
            # For consistency in code paths, make it indexable like a list
            if isinstance(self.k_symbols, sp.Symbol):
                self.k_symbols = [self.k_symbols]
        else:
            # numeric path: you still need the length for checks
            self.k_symbols = [None] * self.d

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
        # convert to backend-specific matrix/array
        H_k = sp.Matrix(H_k) if self.symbolic else np.asarray(H_k, dtype=complex)
        if H_k.shape != (self.N, self.N):
            raise ValueError(f"H(k) must be {self.N}x{self.N}, got {H_k.shape}.")

        if self.symbolic:
            G_inv = (self.omega + self.q * self.eta * sp.I) * self.I - H_k
            G_k = invert_matrix(G_inv, symbolic=True)
        else:
            G_inv = (self.omega + self.q * self.eta * 1j) * self.I - H_k
            G_k = invert_matrix(G_inv, symbolic=False)

        if self.verbose:
            print("\nComputing Green's function at k:")
            print("\nwith k = ", momentum)
            print_symbolic_matrix(
                H_k, name="H(k)") if self.symbolic else print("H(k) =\n", H_k)
            print_symbolic_matrix(G_inv, name="( ω ± iη - H(k) )") if self.symbolic else print(
                "Inversion target =\n", G_inv)
        
        return G_k

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
        # 1) momentum validation
        if len(momentum) != self.d:
            raise ValueError(f"Expected {self.d} momentum components, got {len(momentum)}.")

        # 2) H(k) build + shape check
        H_k = self.H(momentum)
        if self.symbolic:
            H_k = sp.Matrix(H_k)
        else:
            H_k = np.asarray(H_k, dtype=complex)

        if H_k.shape != (self.N, self.N):
            raise ValueError(f"H(k) must be {self.N}x{self.N}, got {H_k.shape}.")

        # 3) G^{-1}(k)
        if self.symbolic:
            G_inv = (self.omega + self.q * self.eta * sp.I) * self.I - H_k
        else:
            G_inv = (self.omega + self.q * self.eta * 1j) * self.I - H_k

        # 4) Eigendecomposition
        if self.symbolic:
            evects = G_inv.eigenvects()
            pairs = []
            for lam, mult, vecs in evects:
                for v in vecs:
                    v = sp.Matrix(v)
                    if v.norm() != 0:
                        v = v / v.norm()
                    pairs.append((sp.simplify(lam), v))

            if len(pairs) != self.N:
                raise ValueError("G^{-1}(k) is not diagonalizable: insufficient eigenvectors.")

            def _lam_key(l):
                return (sp.re(l), sp.im(l))
            # Sorting by real part first, imaginary part second gives a fixed, reproducible order
            pairs.sort(key=lambda t: _lam_key(t[0]))

            eigenvalues = [lam for lam, _ in pairs]
            P = sp.Matrix.hstack(*[v for _, v in pairs])

            try:
                P_inv = P.inv()
            except Exception:
                raise ValueError("G^{-1}(k) is not diagonalizable: eigenbasis is singular.")

            D = sp.diag(*eigenvalues)

            # --- Consistency check: D_direct (from eigenvalues) vs D_reconstructed (P^{-1} G^{-1} P)
            D_recon = sp.simplify(P_inv * G_inv * P)

            # (1) Off-diagonals must vanish exactly (or simplify to zero)
            offdiag = D_recon - sp.diag(*[D_recon[i, i] for i in range(self.N)])
            if not offdiag.equals(sp.zeros(self.N)):
                if not sp.simplify(offdiag).is_zero_matrix:
                    raise ValueError("Eigendecomposition inconsistency: off-diagonal terms remain in P^{-1} G^{-1} P.")

            # (2) Diagonals must match eigenvalues (after simplification)
            for i in range(self.N):
                diff = sp.simplify(D_recon[i, i] - D[i, i])
                # equals(0) can be None; also try is_zero/simplify to be safe
                if not (diff.equals(0) or diff.is_zero):
                    raise ValueError("Eigendecomposition inconsistency: diagonal of P^{-1} G^{-1} P does not match eigenvalues.")

            if self.verbose:
                print("\nDiagonal elements (eigenvalues) of G⁻¹(k):")
                for i, lam in enumerate(eigenvalues):
                    print(f"λ[{i}] = {sp.simplify(lam)}")

            residual = (P * D * P_inv - G_inv)
            if not residual.equals(sp.zeros(self.N)):
                if not sp.simplify(residual).is_zero_matrix:
                    raise ValueError("Eigendecomposition failed: P*D*P^{-1} != G^{-1}(k).")

            return P, eigenvalues, D

        else:
            vals, vecs = np.linalg.eig(G_inv)
            # Sorting by real part first, imaginary part second gives a fixed, reproducible order
            idx = np.lexsort((vals.imag, vals.real))
            vals = vals[idx]
            vecs = vecs[:, idx]

            P = np.array(vecs, dtype=complex)
            condP = np.linalg.cond(P)
            # optional warning:
            if not np.isfinite(condP) or condP > 1e12:
                warnings.warn(f"Ill-conditioned eigenbasis (cond={condP:.2e}). Results may be unstable.")

            P_inv = invert_matrix(P, symbolic=False)
            D = np.diag(vals.astype(complex))

            # --- Consistency check: D_direct vs D_reconstructed (up to numerical error)
            D_recon = P_inv @ G_inv @ P

            # (1) Off-diagonal energy should be ~0
            offdiag = D_recon.copy()
            np.fill_diagonal(offdiag, 0.0)
            offdiag_norm = np.linalg.norm(offdiag)
            if offdiag_norm > NUM_EIG_TOL:
                raise ValueError(
                    f"Eigendecomposition inconsistency: off-diagonal norm {offdiag_norm:.2e} exceeds {NUM_EIG_TOL:.1e}"
                )

            # (2) Diagonal entries should match eigenvalues within tolerance (order already enforced by sorting)
            diag_diff = np.diag(D_recon) - np.diag(D)
            rel_err = np.linalg.norm(diag_diff) / max(1.0, np.linalg.norm(np.diag(D)))
            if rel_err > NUM_EIG_TOL:
                raise ValueError(
                    f"Eigendecomposition inconsistency: diagonal mismatch rel. error {rel_err:.2e} exceeds {NUM_EIG_TOL:.1e}"
                )

            if self.verbose:
                print("\nDiagonal elements (eigenvalues) of G⁻¹(k):")
                for i, lam in enumerate(vals):
                    print(f"λ[{i}] = {lam}")

            num = np.linalg.norm(P @ D @ P_inv - G_inv)
            den = max(1.0, np.linalg.norm(G_inv))
            if num / den > NUM_EIG_TOL:
                raise ValueError("Eigendecomposition failed: reconstruction error above tolerance.")

            return P, vals, D

    def compute_roots_greens_inverse(self, solve_for: int):
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
        solve_for : int
        Index of the momentum component to solve for: 0..(d-1)

        Returns
        -------
        List[Tuple[str, sp.Set]]
            A list of (label, solution_set) pairs, one per eigenvalue λ_i, where the set contains the
            solutions for k_{solve_for} in the complex domain. Non-polynomial cases may return a ConditionSet.

        Raises
        ------
        TypeError
            If called in numeric mode (symbolic=False) or if `solve_for` is not an int.
        ValueError
            If `solve_for` is out of range [0, d-1].
        """
        if not self.symbolic:
            raise TypeError("Root solving is only supported in symbolic mode. Enable symbolic=True.")
        
        # validate index
        if not isinstance(solve_for, int):
            raise TypeError(f"'solve_for' must be an int in [0, {self.d-1}] (k-dimension index).")
        if solve_for < 0 or solve_for >= self.d:
            valid_indices = ", ".join(str(i) for i in range(self.d))
            raise ValueError(f"'solve_for' out of range: got {solve_for}, valid indices are {{{valid_indices}}}.")

        k_var = self.k_symbols[solve_for] # variable to solve for

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

            if not lambda_i.is_polynomial(k_var):
                warnings.warn(
                    f"Solving λ_{i}(k) = 0 may fail: expression is not polynomial in {k_var}.", stacklevel=2)

            try:
                lam_simpl = sp.simplify(lambda_i)
                try:
                    # polynomial attempt
                    poly = sp.Poly(lam_simpl, k_var, domain=sp.CC)
                    if poly.total_degree() > 0:
                        roots = poly.all_roots()
                        solset = sp.FiniteSet(*roots)
                    else:
                        solset = sp.S.Complexes if lam_simpl == 0 else sp.EmptySet
                except sp.PolynomialError:
                    # general solve
                    solset = sp.solveset(sp.Eq(lam_simpl, 0), k_var, domain=sp.S.Complexes)
                    if isinstance(solset, sp.ConditionSet):
                        warnings.warn(
                            f"λ[{i}] is not polynomial in {k_var}; returning a ConditionSet.",
                            stacklevel=2
                        )
                root_solutions.append((f"lambda_{i}=0", solset))
            except Exception as e:
                # fallback if something really unexpected happens
                root_solutions.append((f"lambda_{i}=0", f"Error during solving: {e}"))

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
        - z, z′: Real numbers or real symbols; coordinates along the last spatial dimension.
        - full_matrix: If True, reconstruct the full Green's function matrix in its original basis (not just the diagonal form).

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
            z_diff = z - z_prime
            sig = int(sp.sign(z_diff))
            if sig == 0:
                sig = self.q  # default to retarded(+)/advanced(-) choice
            z_diff_sign = sig

        else:
            assert isinstance(
                z_prime, sp.Symbol), "Expected z' to be instance of sp.Symbol since z is one."

        if self.verbose:
            print("( ω ± iη - H(k) ) will be diagonalized to evaluate residues for the Fourier integral.")
        
        _, eigenvalues, _ = self.compute_eigen_greens_inverse(kvec)

        G_z_diag = [] # List for the diagonal entries of G(z,z'), each the solution of an integral
        has_contributions = False

        for i, lambda_i in enumerate(eigenvalues):
            contrib, contributed_any = self._residue_sum_for_lambda(lambda_i, z, z_prime, k_dir, z_diff_sign)
            has_contributions = has_contributions or contributed_any
            G_z_diag.append(contrib if contributed_any else 0)

            if self.verbose:
                print(f"\nλ_{i}(k) = {lambda_i}")
                print(f"  Contribution to residue sum: {contrib}")

        if not has_contributions:
            warnings.warn("No poles passed the sign check; returning zero Green's function.")
            return sp.zeros(len(self.I))

        if all((val.is_zero is True) for val in G_z_diag):
            warnings.warn(
                "Green's function is identically zero: all residue contributions canceled.")

        G_z = sp.diag(*G_z_diag)
        # Note: Currently returning only diagonal Green's function G(z, z′)
        # Full matrix reconstruction from eigenbasis can be added if needed:
        if full_matrix:
            eigenbasis, _, _ = self.compute_eigen_greens_inverse(kvec)
            G_full = eigenbasis @ G_z @ invert_matrix(eigenbasis, symbolic=True)
            return G_full

        return G_z
    
    def compute_rspace_greens_symbolic_1d(self, 
                                          z: Union[float, sp.Basic],
                                          z_prime: Union[float, sp.Basic],
                                          full_matrix: bool = False):
        # Wrapper for 1D real-space Green's function computation
        # 1D refers to last dimension
        return self.compute_rspace_greens_symbolic_1d_along_last_dim(z, z_prime, full_matrix)
    
    def compute_rspace_greens_numeric_1D(self,
                                         z: float,
                                         z_prime: float,
                                         full_matrix: bool = False): # placeholder function for later implementation
        if self.symbolic:
            warnings.warn(
                "Numeric 1D G(z,z') computation is not supported in symbolic mode. Disable: symbolic=False.")
            
        warnings.warn("Numeric 1D G(z,z') not implemented yet; returning [].")
        return []

    # --- Internal utilities ---
    
    def _residue_sum_for_lambda(self, lambda_i, z, z_prime, kz_sym, z_diff_sign):
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
        - contrib: Total residue contribution to G_{ii}(z, z′)
        - has_contributions: Updated flag
        """
        contributed_any = False
        delta_z = z - z_prime
        phase = sp.exp(sp.I * kz_sym * delta_z)

        if not lambda_i.is_polynomial(kz_sym):
            # Not polynomial (SymPy can’t reliably find poles)
            # Triggers unevaluated integral fallback
            warnings.warn(f"Eigenvalue is not polynomial in k; returning unevaluated Fourier integral.")
            expr = sp.Integral(phase / sp.simplify(lambda_i),
                            (kz_sym, -sp.oo, sp.oo)) / (2*sp.pi)
            contributed_any = True
            if self.verbose:
                print(f"  Unevaluated integral expression for G(k) diagonal entry: {expr}")
            return expr, contributed_any 

        poly = sp.Poly(sp.simplify(lambda_i), kz_sym, domain=sp.CC)
        roots_with_mult = poly.roots()  # dict: {root: multiplicity}
        if self.verbose:
            print(f"  Found roots (with multiplicity) of eigenvalue {lambda_i}: {roots_with_mult}")
        
        contrib = 0
        for k0, m in roots_with_mult.items(): # roots k0 with their multiplicity m
            # Half-plane selector
            if z_diff_sign != sp.sign(sp.im(k0)):
                if self.verbose:
                    print(f"Pole k={k0} (multiplicity={m}) lies in wrong half-plane; skipped.")
                continue

            # Residue formula for pole of order m:
            # Res = 1/(m-1)! * d^{m-1}/dk^{m-1} [ (k-k0)^m * phi / lambda_i(k) ] at k=k0
            expr = sp.simplify(((kz_sym - k0)**m) * phase / lambda_i)
            if m == 1:
                res = sp.simplify(expr.subs(kz_sym, k0))  # zero-th derivative
            else:
                deriv = sp.diff(expr, (kz_sym, m - 1))
                res = sp.simplify(deriv.subs(kz_sym, k0) / sp.factorial(m - 1))

            contrib += sp.I * res # factor of i from residue theorem
            contributed_any = True
            
        return contrib, contributed_any

