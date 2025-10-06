# region Imports & metadata
"""
Green’s function utilities: k-space and real-space calculations.
Both symbolic (SymPy) and numeric (NumPy) backends supported.

Contents
--------
- GreensFunctionCalculator : main public class
- (internal) _helpers      : implementation details

Usage
-----
>>> G = GreensFunctionCalculator(...).compute_kspace_greens_function(k)

Notes
-----
- Logging: this library emits logs; your runner configures handlers.
- Ambiguity: see AmbiguityLedger for collecting/formatting cases.
             more info in docs/ambiguity.md

Public exports
--------------
__all__ = ["GreensFunctionCalculator"]
"""
import numpy as np
import sympy as sp
from sympy import pprint
from typing import Callable, Union, Sequence
from utils import invert_matrix, sanitize_vector, sanitize_matrix
import warnings
from ambiguity import AmbiguityLedger, AggregatedAmbiguityError
import logging

__all__ = ["GreensFunctionCalculator"]
# endregion

# region Constants & module-level config
log = logging.getLogger(__name__)

MatrixLike = Union[np.ndarray, sp.Matrix]
ArrayLike  = Union[Sequence[float], np.ndarray, sp.Matrix]

NUM_EIG_TOL = 1e-8 # reconstruction tolerance for eigen-decomp checks
INFINITESIMAL = 1e-6  # default infinitesimal if none provided
# endregion

# region GreensFunctionCalculator (public class)
class GreensFunctionCalculator:
    # region Construction & dunder methods
    def __init__(self,
                 hamiltonian: Callable[[ArrayLike], MatrixLike],
                 identity: MatrixLike,
                 symbolic: bool,
                 # omega
                 energy_level: Union[float, sp.Basic],
                 # eta
                 broadening: Union[float, sp.Basic] = None,
                 # defaults to retarded Green's functions
                 retarded: bool = True,
                 dimension: int = 3):
        """
        Calculator for k-space and real-space Green’s functions.

        Public API (stable)
        -------------------
        - compute_kspace_greens_function(k) -> MatrixLike
        - compute_roots_greens_inverse(solve_for=None) -> list[RootInfo]
        - compute_rspace_greens_symbolic_1d(x, ...) -> Expr | Array
        - __repr__(), __str__()

        Parameters
        ----------
        hamiltonian: Callable[[ArrayLike], MatrixLike]
            a function that takes momentum k and returns the NxN Hamiltonian matrix
        identity: MatrixLike
            identity matrix (NxN) for the appropriate backend, where N is the band size
        symbolic: Boolean
            whether to use SymPy as backend (symbolic=True) or NumPy (symbolic=False)
        energy_level: Float or sp.Symbol
            scalar ω
        broadening: Float or sp.Symbol, positive
            small η > 0 to define the imaginary part
        retarded: Boolean
            if True computes retarded Green's function; else advanced
        dimension: Int
            spatial dimension of the system (1, 2, or 3), defaults to 3

        Notes
        -----
        Private helpers are underscore-prefixed and may change without notice.
        See also: `get_ambiguities()`, `format_ambiguities()` for diagnostics.
        """
        self.H = hamiltonian
        if not callable(self.H):
            raise ValueError(
                "Hamiltonian must be a callable function H(k).")
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
                    "Broadening η is zero; Green's function may be ill-defined at poles.", stacklevel=2)
        elif self.symbolic:
            self.eta = sp.symbols("eta", positive=True)
            warnings.warn("No broadening η provided; using symbolic η > 0.", stacklevel=2)
        else:
            self.eta = INFINITESIMAL
            warnings.warn(
                f"No broadening η provided; defaulting to η={self.eta}.", stacklevel=2)
            
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

        self.green_type = "retarded (+iη)" if self.q == 1 else "advanced (−iη)"

        log.debug("Initialized %r", self) # developer snapshot for logs
        log.info("Snapshot:\n%s", self) # readable banner for operators/notebooks
        self._ledger = AmbiguityLedger()
    
    def __repr__(self):
        try:
            mode = "sym" if self.symbolic else "num"
            I_shape = getattr(self.I, "shape", None)
            I_summary = f"{type(self.I).__name__}{I_shape}" if I_shape is not None else type(self.I).__name__
            H_name = getattr(self.H, "__name__", None) or type(self.H).__name__
            try:
                if isinstance(self.k_symbols, (list, tuple)) and self.k_symbols:
                    k_summary = ",".join(str(s) if s is not None else "∅" for s in self.k_symbols)
                else:
                    k_summary = "∅"
            except Exception:
                k_summary = "∅"
            return (f"{self.__class__.__name__}("
                    f"mode={mode}, N={self.N}, d={self.d}, "
                    f"ω={self.omega}, η={self.eta}, type={self.green_type}, "
                    f"I={I_summary}, H={H_name}, k={k_summary})")
        except Exception:
            return f"{self.__class__.__name__}(unprintable; id=0x{id(self):x})"


    def __str__(self):
        try:
            mode = "symbolic" if self.symbolic else "numeric"
            I_shape = getattr(self.I, "shape", None)
            identity_summary = f"{type(self.I).__name__}{I_shape}" if I_shape is not None else type(self.I).__name__
            # k symbols line (optional)
            k_line = ""
            try:
                ks = self.k_symbols
                if isinstance(ks, (list, tuple)) and ks:
                    items = []
                    for s in ks[:8]:
                        try:
                            items.append(str(s) if s is not None else "∅")
                        except Exception:
                            items.append("<unprintable>")
                    suffix = " …" if len(ks) > 8 else ""
                    k_line = f"\n  k symbols: {', '.join(items)}{suffix}"
            except Exception:
                k_line = ""
            return (
                "GreensFunctionCalculator\n"
                f"  mode: {mode}\n"
                f"  N×N: {self.N}×{self.N}   d: {self.d}\n"
                f"  ω: {self.omega}   η: {self.eta}   type: {self.green_type}\n"
                f"  identity: {identity_summary}"
                f"{k_line}"
            )
        except Exception:
            return f"{self.__class__.__name__} (unprintable)"
    # endregion

    # region k-space Green’s function
    def compute_kspace_greens_function(self, momentum: ArrayLike | None = None) -> MatrixLike:
        """
        Compute the Green's function for a single-particle Hamiltonian in momentum space by inverting
        (omega + q*i*eta - H(k)), where q = ±1 for retarded/advanced GF.

        Parameters
        ----------
        momentum: ArrayLike or None
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
        self._reset_ambiguities()
        log.debug("Computing G(k) with: momentum=%s (symbolic=%s)", momentum, self.symbolic)

        if momentum is None:
            if self.symbolic:
                momentum = sp.Matrix(self.k_symbols)
            else:
                raise ValueError(
                    "Momentum must be provided in numeric mode (symbolic=False).")
        k_vec = sanitize_vector(momentum, self.symbolic, expected_dim=self.d) # ensure iterable and correct type and shape
        k_for_H = (k_vec[0] if self.d == 1 else k_vec) # scalar only for H(k), since H expects scalar, if d=1
            
        log.debug("Calling H(k) for %s; k=%s", getattr(self.H, "__name__", type(self.H).__name__), k_for_H)  
        H_k = self.H(k_for_H)  # Hamiltonian at momentum k
        H_k = sanitize_matrix(H_k, self.symbolic, expected_size=self.N)
        log.debug("H(k) built, shape=%s", H_k.shape)

        imaginary_unit = sp.I if self.symbolic else 1j
        G_inv = (self.omega + self.q * self.eta *
                 imaginary_unit) * self.I - H_k
        log.debug("Formed G^{-1}(k) = (ω %s iη)I - H(k)", "+" if self.q==1 else "-")
        G_k = invert_matrix(G_inv, symbolic=self.symbolic)
        log.info("G(k) computed successfully.") 
        log.debug("G(k): shape=%s, backend=%s", getattr(G_k,"shape",None), "sym" if self.symbolic else "num")     

        return G_k
    # endregion

    # region Root solving
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
        log.debug("Started root solving")
        self._reset_ambiguities()
        
        if not self.symbolic:
            warnings.warn(
                "Root solving is only supported in symbolic mode. Enable symbolic=True.", stacklevel=2)
            return []  # no error raised but empty list returned
        
        # optional: a list of SymPy assumptions to resolve ambiguous points for the solver
        predicates, choices = self._split_case_assumptions(case_assumptions)

        if solve_for is None:
            solve_for = self.d - 1  # default to last dimension
            log.debug("No solve_for input provided. Defaulting to solve_for=%d", solve_for)
        # validate index
        if not isinstance(solve_for, int):
            raise TypeError(
                f"'solve_for' must be an int in [0, {self.d-1}] (k-dimension index).")
        if solve_for < 0 or solve_for >= self.d:
            valid_indices = ", ".join(str(i) for i in range(self.d))
            raise ValueError(
                f"'solve_for' out of range: got {solve_for}, valid indices are {{{valid_indices}}}.")

        k_var = self.k_symbols[solve_for]  # variable to solve for
        log.info("Computing roots of G⁻¹(k) for variable %s.", k_var)

        # Define symbolic momentum components
        k = sp.Matrix(self.k_symbols)  # e.g., Matrix([k_x, k_y, k_z]) or fewer

        with sp.assuming(*predicates):
            # Compute eigenvalues of the inverse Green's function
            _, eigenvalues, _ = self._eigenvalues_greens_inverse(k)
            log.debug("Eigenvalues successfully computed.")

            root_solutions = []
            non_empty_flag = 0  # count how many eigenvalues have non-empty solution sets
            for i, lambda_i in enumerate(eigenvalues):
                # simplify for readability and solving
                lambda_i = sp.simplify(sp.together(lambda_i))
                # a) Short circuit if λ_i has no dependence on k_var
                if not lambda_i.has(k_var):
                    #log.debug("Free symbols of λ_%d: %s", i, lambda_i.free_symbols)
                    # check for all-together vanishing eigenvalue
                    if lambda_i.equals(0) or lambda_i.is_zero:
                        log.error("Eigenvalue λ_%d is identically zero; G⁻¹(k) singular.", i)
                        raise ValueError(
                            f"Eigenvalue λ_{i} is identically zero for all {k_var}, indicating a singular G⁻¹(k).")
                    # λ_i is constant in the variable to solve for
                    warnings.warn(
                        f"Eigenvalue lambda_{i} is constant in k_var; returning empty solution set.", stacklevel=2)
                    # Return empty solution set
                    root_solutions.append((f"lambda_{i}=0", sp.EmptySet))
                    continue

                # b) Solve for roots of lambda_i = 0
                try:
                    try:
                        # polynomial attempt
                        # let SymPy pick the domain
                        log.debug("Trying to solve polynomially.")
                        poly = sp.Poly(lambda_i, k_var)
                        if poly.total_degree() > 0:
                            # dict {root: multiplicity}
                            roots_dict = sp.roots(poly.as_expr(), k_var)
                            # expand multiplicities into a list, to match your previous FiniteSet(*roots)
                            roots_list = []
                            for r, m in roots_dict.items():
                                roots_list.extend([sp.simplify(r)] * int(m))
                            solset = sp.FiniteSet(*roots_list)
                        else:
                            log.debug("Polynomial degree is zero.")
                            log.debug("Short circuit should have caught this case earlier.")
                            # check for all-together vanishing eigenvalue
                            expr = poly.as_expr()
                            if expr.has(k_var):
                                # strange edge case that needs to be resolved if encountered
                                choice_key = ("roots", f"lambda_{i}.poly_constant_yet_not")
                                if choices[choice_key] != "constant":
                                    self._add_amb(
                                        where="roots",
                                        what=f"lambda_{i}.poly_constant_yet_not",
                                        predicate=None,                     # no simple predicate; depends on k0
                                        options=["constant", "not-constant"],
                                        consequence=f"Cannot decide if λ_{i}({k_var})=0 has roots.",
                                        data={"k_var": k_var, "lambda_i": lambda_i, "poly_expr": expr},
                                        severity="error"  # unresolved unless a choice or predicate resolves it
                                    )
                                    root_solutions.append(
                                    (f"lambda_{i}=0", f"Edge case: polynomial has {k_var} but degree is zero. Solver failed."))
                                    continue
                                log.debug(f"{choice_key[1]} ambiguity resolved by choice provided.")
                                
                            # check for all-together vanishing eigenvalue
                            if expr.equals(0) or expr.is_zero:
                                log.error("Eigenvalue λ_%d is identically zero; G⁻¹(k) singular.", i)
                                raise ValueError(
                                    f"Eigenvalue λ_{i} is identically zero for all {k_var}, indicating a singular G⁻¹(k).")
                            # λ_i is constant in the variable to solve for
                            warnings.warn(
                                f"Eigenvalue lambda_{i} is constant in k_var; returning empty solution set.", stacklevel=2)
                            # Return empty solution set
                            root_solutions.append((f"lambda_{i}=0", sp.EmptySet))
                            continue
                    except sp.PolynomialError:
                        # general solve for non-polynomial case
                        log.debug("Polynomial solver failed. Trying general solver with sp.solveset.")
                        warnings.warn(f"Eigenvalue λ_{i} is not polynomial in {k_var}")
                        solset = sp.solveset(
                            sp.Eq(lambda_i, 0), k_var, domain=sp.S.Complexes)
                        if isinstance(solset, sp.ConditionSet):
                            choice_key = ("roots", f"lambda_{i}.condition_set")
                            if choices[choice_key] != "ConditionSet":
                                self._add_amb(
                                    where="roots",
                                    what=f"lambda_{i}.condition_set",
                                    predicate=None,   # no simple predicate; depends on k0
                                    options=["ConditionSet", "FiniteSet"],
                                    consequence=f"Cannot solve λ_{i}(k) = 0. Returning ConditionSet.",
                                    data={"k_var": k_var, "lambda_i": lambda_i},
                                    severity="warn"  # try to resolve with a predicate or assumption
                                )
                                log.debug("Choose 'ConditionSet' for choices[%s] or provide predicate to solve %s=0", choice_key, lambda_i)
                    root_solutions.append((f"lambda_{i}=0", solset))
                except Exception as e:
                    # fallback if something really unexpected happens
                    log.error("Error during solving for lambda_%d: %s", i, e)
                    root_solutions.append(
                        (f"lambda_{i}=0", f"Error during solving."))
                non_empty_flag = 1

            if non_empty_flag == 0:
                warnings.warn(
                    "None of the eigenvalues depend on k_var; G⁻¹(k) has no roots.", stacklevel=2)
            log.info("Root solving completed.")
            log.debug("Roots of G⁻¹(k) %s", root_solutions)
            self._finalize_ambiguities_or_raise(context="root solving")

        return root_solutions
    # endregion

    # region Real-space Fourier transform
    # -- Symbolic 1D real-space transform --
    def compute_rspace_greens_symbolic_1d_along_last_dim(self,
                                                         z: float | sp.Basic,
                                                         z_prime: float | sp.Basic,
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
        log.debug("Started 1D real-space transform of the %s Green's function.", self.green_type)
        self._reset_ambiguities()

        # optional: a list of SymPy assumptions to resolve ambiguous points for the solver
        assumptions_made = False
        if case_assumptions is not None: 
            assumptions_made = True
        predicates, choices = self._split_case_assumptions(case_assumptions)

        if not self.symbolic:
            warnings.warn(
                "Symbolic 1D G(z,z') computation is only supported in symbolic mode. Enable symbolic=True. Returning empty list.", stacklevel=2)
            return []  # no error raised but empty list returned with warning

        if z_diff_sign is None and self._halfplane_choice(z, z_prime) is None:
            raise ValueError(
                f"No z_diff_sign provided and sign({z} - {z_prime}) indeterminable. Provide z_diff_sign or exchange {z},{z_prime} for numeric values.")
        else: 
            z_diff_sign = self._halfplane_choice(z, z_prime) if z_diff_sign is None else z_diff_sign
            log.debug("Halfplane choice: z_diff_sign=%d.", z_diff_sign)

        kvec = sp.Matrix(self.k_symbols)
        # direction of real-space transform (last component)
        k_dir = self.k_symbols[self.d - 1]
        log.debug("Integration coordinate: k_dir=%s.", k_dir)

        z_sym, zp_sym = sp.sympify(z), sp.sympify(z_prime)
        if z_sym.is_real is False or zp_sym.is_real is False:
            raise ValueError("Both z and z′ must be real symbols or real numbers.")

        log.debug("Real space coordinates: %s, %s.", z_sym, zp_sym)

        if not isinstance(z, sp.Symbol):
            if isinstance(z_prime, sp.Symbol):
                raise TypeError("Expected z' to be a real number like z, not a symbol.")
            z = float(z)
            z_prime = float(z_prime)
            z_diff = z - z_prime
            sig = int(sp.sign(z_diff))
            if sig == 0:
                warnings.warn("z and z' are equal; results may be singular.", stacklevel=2)
            if not z_diff_sign == sig: 
                raise ValueError(
                    f"Expected the sign of (z-z')={z-z_prime} to match z_diff_sign={z_diff_sign}. Change to z_diff_sign=None for automatic alignment or match z_diff_sign={sig}.")

        else:
            if not isinstance(z, sp.Symbol):
                raise TypeError(f"Expected z' to be a symbol like z, got {type(z_prime)}.")

        if self.green_type == "retarded (+iη)":
            log.debug("Diagonalizing (ω + iη - H(k)) to evaluate residues for the Fourier integral.") 
        else:
            log.debug("Diagonalizing (ω - iη - H(k)) to evaluate residues for the Fourier integral.")

        eigenbasis, eigenvalues, _ = self._eigenvalues_greens_inverse(kvec)
        log.debug("Eigenvalues successfully computed.")
        log.info(
            "The Fourier integral over %s is computed using contour integration.", k_dir) 
        if z_diff_sign == 1:
            log.info("The contour is closed in the upper half-plane, since z>z'.")
        elif z_diff_sign == -1:
            log.info("The contour is closed in the lower half-plane, since z<z'.")
        log.info("If different assumptions on %s and %s are needed, rerun with the appropriate z_diff_sign parameter.", z, z_prime)

        # List for the diagonal entries of G(z,z'), each the solution of an integral
        G_z_diag = []
        has_contributions = False # track if any pole contributed

        for i, lambda_i in enumerate(eigenvalues):
            contrib, contributed_any = self._residue_sum_for_lambda(
                i, lambda_i, z, z_prime, k_dir, z_diff_sign, predicates, choices)
            has_contributions = has_contributions or contributed_any
            assert contrib == 0 if not contributed_any else True, "If no poles contributed, the contribution must be zero."
            G_z_diag.append(contrib)

            log.debug("Processed eigenvalue λ_%d=%s.", i, lambda_i)
            log.debug("Diagonal entry to G(z,z') #%d: %s", i, contrib)
        self._finalize_ambiguities_or_raise(context="root solving")   

        if not has_contributions:
            warnings.warn(
                "No poles passed the sign check; returning zero Green's function.", stacklevel=2)
            assert sp.diag(*G_z_diag) == sp.zeros(self.N), "Expected zeros on the diagonal if no poles contributed."

        elif all((val.is_zero is True) for val in G_z_diag):
            warnings.warn(
                "Green's function is identically zero: all residue contributions canceled out.", stacklevel=2)
            assert has_contributions == True, "Expected has_contributions=True if some poles contributed, otherwise earlier warning should have been triggered."

        G_z = sp.diag(*G_z_diag)
        log.info("1D real-space %s Green's function G(%s,%s) successfully computed.", self.green_type, z, z_prime)
        if assumptions_made == True:
            log.info(f"Note: Additional assumptions were made. Keep in mind.") 
            log.debug("Case assumptions: %s", case_assumptions)

        # Note: Currently returning only diagonal Green's function G(z, z′)
        # Full matrix reconstruction from eigenbasis can be added if needed:
        if full_matrix:
            G_full = eigenbasis @ G_z @ invert_matrix(
                eigenbasis, symbolic=True)
            log.info("The full Green's function matrix in the original basis is returned.")
            return G_full
            
        log.info("Note: Only diagonal entries are returned by default.") 
        return G_z

    def compute_rspace_greens_symbolic_1d(self, z, z_prime, full_matrix: bool = False):
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
        z_diff_sign = 1 # default for test purposes
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
                "Numeric 1D G(z,z') computation is not supported in symbolic mode. Disable: symbolic=False.", stacklevel=2)

        warnings.warn("Numeric 1D G(z,z') not implemented yet; returning [].", stacklevel=2)
        return []
    # endregion

    # region Ambiguity helpers 
    def _reset_ambiguities(self): 
        self._ledger.reset()
        log.debug("Ambiguity ledger reset.")
    def _add_amb(self, **kw): self._ledger.add(**kw)
    def _split_case_assumptions(self, case_assumptions):
        # returns (predicates: list[sp.Basic], choices: dict)
        if case_assumptions is None:
            log.debug("No case_assumptions provided.")
            return [], {}
        if isinstance(case_assumptions, list):
            log.debug("case_assumptions provided as list of predicates.")
            return case_assumptions, {}
        if isinstance(case_assumptions, dict):
            preds = case_assumptions.get("predicates", [])
            choices = case_assumptions.get("choices", {})
            log.debug("case_assumptions provided as dict with %d predicates and %d choices.", len(preds), len(choices))
            return preds, choices
        raise TypeError("case_assumptions must be list or dict with 'predicates'/'choices'.")
    def _finalize_ambiguities_or_raise(self, context: str):
        items = self.get_ambiguities()
        if not items:
            return
        # escalate if any is 'error'
        has_error = any(a.severity == "error" for a in items)
        summary = self.format_ambiguities()
        if has_error:
            log.error("Ambiguities escalated to error during %s.", context)
            raise AggregatedAmbiguityError(
                f"Ambiguities encountered during {context}:\n{summary}", items=items)
        else:
            log.warning("Ambiguities encountered during %s:\n%s", context, summary)
    def get_ambiguities(self): return self._ledger.items()
    def format_ambiguities(self): return self._ledger.format()
    # endregion

    # region Internal utilities

    @staticmethod
    def _halfplane_choice(z, z_prime):
        # numeric-only decision; returns +1, -1, 0, or None (unknown)
        z, z_prime = sp.sympify(z), sp.sympify(z_prime)
        if z.is_real is not True or z_prime.is_real is not True:
            raise ValueError("Both z and z′ must be real numbers or real symbols.")
        if z.is_number and z_prime.is_number:
            if z > z_prime:  return +1
            if z < z_prime:  return -1
            return 0
        return None
    
    def _im_sign_of_root(self, k0, eta_symbol=None, predicates=None):
        """
        Try to determine sign(Im(k0)) robustly.
        Returns +1, -1, or None if undecidable.
        """
        predicates = predicates or []

        # 1) Direct attempt under assumptions
        with sp.assuming(*predicates):
            s = sp.sign(sp.im(k0))
            if s in (sp.Integer(1), sp.Integer(-1)):
                return int(s)
            if s == 0:
                return 0

        # 2) η -> 0+ limit (if an eta symbol is known/used)
        if eta_symbol is None:
            # try to infer your η
            eta_symbol = getattr(self, "eta", None)
            if not isinstance(eta_symbol, sp.Symbol):
                eta_symbol = None
        if isinstance(eta_symbol, sp.Symbol) and k0.has(eta_symbol):
            with sp.assuming(*predicates):
                try:
                    lim_im = sp.limit(sp.im(k0), eta_symbol, 0, dir='+')
                    s = sp.sign(sp.simplify(lim_im))
                    if s in (sp.Integer(1), sp.Integer(-1)):
                        return int(s)
                    if s == 0:
                        return 0
                except Exception:
                    pass

        # 3) sqrt(a + I*b) pattern
        try:
            base = k0
            print("base = ", base)
            print("base.func: ", base.func)
            base_squared = sp.simplify(sp.together(k0**2))
            print("base_squared = ", base_squared)
            print("base_squared.func = ", base_squared.func)
            sign_flip = 1
            if base.func is sp.Mul:
                # factor out explicit -1 if present
                coeffs = [arg for arg in base.args if arg.is_Number]
                print("coeffs =", coeffs)
                if any(c == -1 for c in coeffs):
                    sign_flip = -1
                    base = sp.Mul(*(a for a in base.args if a != -1))
            if base.func is sp.sqrt:
                expr = base.args[0]
                with sp.assuming(*predicates):
                    b = sp.im(expr)
                    s = sp.sign(sp.simplify(b))
                    if s in (sp.Integer(1), sp.Integer(-1)):
                        return sign_flip * int(s)
                    if s == 0:
                        return 0
            b = sp.im(base_squared)
            print("b = ", b)
            s = sp.sign(sp.simplify(b))
            print("s = ", s)
            if s in (sp.Integer(1), sp.Integer(-1)):
                return sign_flip * int(s)
            if s == 0:
                return 0
        except Exception:
            pass
    
    def _eigenvalues_greens_inverse(self, momentum: ArrayLike | None = None) -> tuple[MatrixLike, list[sp.Expr] | np.ndarray, MatrixLike]:

        """
        Diagonalize the inverse Green's function matrix to obtain its eigenbasis and eigenvalues.
        Useful for identifying poles and simplifying root solving.
        Returns eigenbasis P, eigenvalues, and diagonalized matrix D, such that D = P^{-1} G^{-1}(k) P.
        Parameters
        ----------
        momentum: ArrayLike or None
            value at which the Hamiltonian is evaluated
            If None, defaults to k symbols in symbolic mode and raises a ValueError in numeric mode.


        Returns
        ---------
        Tuple (P, eigenvalues, D):
            P : sp.Matrix (symbolic) or np.ndarray (numeric)
            eigenvalues : list[sp.Expr] (symbolic) or np.ndarray (numeric)
            D : sp.Matrix (symbolic) or np.ndarray (numeric)

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
        k_vec = sanitize_vector(momentum, self.symbolic, expected_dim=self.d)   # always sanitize
        k_for_H = (k_vec[0] if self.d == 1 else k_vec)                          # scalar only for H(k)

        # 2) H(k) build + shape check
        H_k = self.H(k_for_H)
        H_k = sanitize_matrix(H_k, self.symbolic, expected_size=self.N)

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
                    f"Ill-conditioned eigenbasis (cond={condP:.2e}). Results may be unstable.", stacklevel=2)

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

            return P, vals, D

    def _residue_sum_for_lambda(self, i, lambda_i, z, z_prime, kz_sym, z_diff_sign, predicates: list, choices: dict):
        """
        Apply the residue theorem to compute the contribution to G(z, z′) from one eigenvalue λᵢ.
        This method of calculating the residue is based on the assumption that the diagonal
        entries λᵢ(k) of G⁻¹(k) are polynomials in the integration variable "kz_sym" (e.g. λᵢ(k_z)= (k_z)^2/(2m)).
        Therefore, G(k) has poles at the roots of λᵢ(k). The multiplicity of those poles are taken into account.
        Only poles in the correct half-plane (sign(im(k0)) == z_diff_sign) contribute.

        Parameters:
        - lambda_i: Diagonal entry λᵢ(k) of G⁻¹(k). Must be polynomial in the integration variable!
        - z, z′: Coordinates in real space (must be real-valued or symbolic real)
        - kz_sym: Momentum variable to integrate over (e.g., k_z)
        - z_diff_sign: Determines correct half-plane for the contour

        Returns:
        - contrib: Total residue contribution to G_{ii}(z, z′), that is the residue sum multiplied by i.
        - contributed_any: Contribution flag
        """
        contributed_any = False

        # Short-circuit if λ_i has no dependence on kz_sym:
        if not lambda_i.has(kz_sym):
            log.debug("Eigenvalue λ_%s does not depend on the integration variable %s.", i, kz_sym)
            log.debug("λ_%s = %s", i, lambda_i)
            contrib = 0
            return contrib, contributed_any

        z_diff = z - z_prime
        phase = sp.exp(sp.I * kz_sym * z_diff)

        with sp.assuming(*predicates):
            if not lambda_i.is_polynomial(kz_sym):
                # Not polynomial (SymPy can’t reliably find poles)
                # Triggers unevaluated integral fallback
                warnings.warn(
                    f"Eigenvalue λ_{i} is not polynomial in {kz_sym}; returning unevaluated Fourier integral.", stacklevel=2)
                expr = sp.Integral(phase / sp.simplify(lambda_i),
                                (kz_sym, -sp.oo, sp.oo)) / (2*sp.pi)
                contributed_any = True
                log.debug("The Residue Theorem could not be applied,")
                log.debug("since λ_%s is not polynomial in %s: %s", i, kz_sym, lambda_i)
                return expr, contributed_any

            # dict {root: multiplicity}
            roots_with_mult = sp.roots(sp.simplify(lambda_i), kz_sym)
            log.debug("Roots of polynomial λ_%s with multiplicities: %s", i, roots_with_mult)

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
                sgn = self._im_sign_of_root(k0, predicates=predicates)
                if sgn in (sp.Integer(1), sp.Integer(-1)):
                    if int(sgn) == z_diff_sign:
                        residue_sum += res
                        contributed_any = True
                    else:
                        log.debug("Pole %s=%s (m=%s) lies in wrong half-plane; skipped.", kz_sym, k0, m)
                elif sgn == 0:
                    # pole lies exactly on the real axis
                    raise ValueError(
                        f"Pole at {kz_sym}={k0} (m={m}) lies on the real axis; integral is ill-defined. Provide finite broadening η.")
                else:
                    choice_key = ("residue", f"lambda_{i}.sign_im_root")
                    if choices.get(choice_key) in (+1, -1):
                        log.debug(f"{choice_key[1]} ambiguity resolved by choice provided.")
                        if choices[choice_key] == z_diff_sign:
                            residue_sum += res
                            contributed_any = True
                        else: 
                            log.debug("Pole %s=%s (m=%s) lies in wrong half-plane; skipped.", kz_sym, k0, m)
                    else:
                        self._add_amb(
                            where="residue",
                            what=f"lambda_{i}.sign_im_root",
                            predicate=None,                     # no simple predicate; depends on k0
                            options=[+1, -1],
                            consequence="Cannot choose contour pole; G(z,z') ambiguous.",
                            data={"k0": k0, "multiplicity": int(m), "lambda_i": lambda_i},
                            severity="error"  # unresolved unless a choice or predicate resolves it
                        )

            contrib = sp.I * residue_sum  # factor of i from residue theorem

        return contrib, contributed_any
    # endregion

# endregion