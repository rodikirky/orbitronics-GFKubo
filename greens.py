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
from sympy import pprint, PolynomialError, ConditionSet
from typing import Callable, Union, Sequence, Optional, Tuple, List
from dataclasses import dataclass
from utils import invert_matrix, sanitize_vector, sanitize_matrix
import warnings
from ambiguity import AmbiguityLedger, AggregatedAmbiguityError
import logging
from func_timeout import func_timeout, FunctionTimedOut

__all__ = ["GreensFunctionCalculator"]
# endregion

# region Constants & module-level config
log = logging.getLogger(__name__)

@dataclass(frozen=True)
class Poly:
    """Container for det(G^{-1}) as a univariate polynomial in the chosen k-component."""
    var: sp.Symbol                 # e.g., k_z
    poly: sp.Poly                  # P(var) = det(G^{-1})(k_var)
    degree: int                    # degree in `var`
    even: bool                     # True if all exponents of `var` are even
    u: Optional[sp.Symbol] = None  # if even: u = var**2
    u_poly: Optional[sp.Poly] = None  # if even: Q(u) with P(var)=Q(var**2)
    free_params: Tuple[sp.Symbol, ...] = ()  # symbols in P other than `var`

@dataclass
class PolyCompiled:
    """
    Backend-compiled numeric callables and metadata for:
      - P(k, *params)
      - Pprime(k, *params)
      - Nij[k, *params] (optional matrix of callables)
    """
    var: sp.Symbol
    params: Tuple[sp.Symbol, ...]
    backend: str
    prec: int # precision hint for the "mpmath" backend
    even: bool
    P: callable
    Pprime: callable
    u: Optional[sp.Symbol] = None
    Q: Optional[callable] = None
    Qprime: Optional[callable] = None

    def args_from_dict(self, vals: dict) -> Tuple:
        """Map a dict of parameter values to the fixed positional order."""
        try:
            return tuple(vals[p] for p in self.params)
        except KeyError as e:
            missing = [p for p in self.params if p not in vals]
            raise KeyError(f"Missing parameter values for: {missing}") from e

MatrixLike = Union[np.ndarray, sp.Matrix]
ArrayLike  = Union[Sequence[float], np.ndarray, sp.Matrix]
#ExprLike = Union[sp.Expr, sp.Poly, Poly] # for some reason this won't be accepted as a type

NUM_TOL = 1e-8 # numerical tolerance for equality checks
DIGITS = 8
INFINITESIMAL = 1e-6  # default infinitesimal if none provided

TIMEOUT_GATE = 12.0 # seconds
# endregion

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

        #log.debug("Initialized %r", self) # developer snapshot for logs
        log.info("Initialized %s", self) # readable banner for operators/notebooks
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

    # region k-space
    def greens_inverse(self, momentum: ArrayLike | None = None) -> MatrixLike:
        '''
        Builds the inverse k-space Green's function as G_inv = (ω +- iη)I - H(k).
        Works in both modes symbolic/numeric. 

        Parameters
        ----------
        momentum: ArrayLike or None
            value at which the Hamiltonian is evaluated
            If None, defaults to k symbols in symbolic mode and raises a ValueError in numeric mode.

        Returns
        -------
        G^{-1}(k): MatrixLike
            Matrix of the shape of the Hamiltonian
            Inverse of the Green's function in momentum space
        '''
        self._reset_ambiguities()
        log.debug("Computing G^{-1}(k) with: momentum=%s (symbolic=%s)", momentum, self.symbolic)

        if self.symbolic:
            momentum = sp.Matrix(self.k_symbols)
        elif momentum is None:
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
        return G_inv
    
    def kspace_greens_function(self, momentum: ArrayLike | None = None) -> MatrixLike:
        """
        Computes the Green's function for a single-particle Hamiltonian in momentum space by inverting
        (omega + q*i*eta - H(k)), where q = ±1 for retarded/advanced GF.
        Works in both modes symbolic/numeric. But full symbolic expression may be very large and slow to evaluate. 

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
        G_inv = self.get_greens_inverse(momentum) if momentum is not None else self.get_greens_inverse()
        G_k = invert_matrix(G_inv, symbolic=self.symbolic)
        log.info("G(k) computed successfully.") 
        log.debug("G(k): shape=%s, backend=%s", getattr(G_k,"shape",None), "sym" if self.symbolic else "num")     
        return G_k
    
    def adjugate_greens_inverse(self, momentum: ArrayLike | None = None) -> MatrixLike:
        '''
        Computes the adjugate of the matrix G_inv.

        Parameters
        ----------
        momentum: ArrayLike or None
            value at which the Hamiltonian is evaluated
            If None, defaults to k symbols in symbolic mode and raises a ValueError in numeric mode.

        Returns
        ---------
        adj(G_inv(k)): MatrixLike
            Adjugate in momentum space
        '''
        G_inv = self.get_greens_inverse(momentum)
        det = self._determinant(G_inv)
        adjugate = G_inv.adjugate() if self.symbolic else det*np.linalg.inv(G_inv)
        return adjugate
    
    def numerator_denominator_poly(self, A: MatrixLike, i: int, j: int, solve_for: int = None) -> tuple[Poly, Poly]: 
        '''
        Selects a matrix entry according to the indeces provided and checks whether that entry has a denominator that depend in the 
        variable to solve for 'k_var'.
        If so, this method returns the numerator and the denominator of the matrix element as polynomials in k_var
        in form of the Poly dataclass.

        Parameters
        ----------
        A: MatrixLike
            Matrix for entry choice
            usually the adjugate of G_inv
        i: int
            Row index
        j: int
            column index
        solve_for: int or None
            identifies the variable, for which the polynomial is formed.
            If None, last dimension is chosen by default.

        Returns
        -------
        num_poly: Poly dataclass
            Matrix entry for given indeces
        den_poly: Poly dataclass
            Denominator as polynomial with metadata

        Raises
        ------
        ValueError
            If called in numeric mode.
        '''
        # 1) Select the chosen entry and identify its denominator
        A_ij = sp.cancel(A[i][j])
        if not self.symbolic:
            raise ValueError("Polynomial cannot be constructed in numeric mode (symbolic = False). Returning matrix entry and None.")
        solve_for = self._clean_solve_for(solve_for, dimension=self.d)
        k_var = self.k_symbols[solve_for]  # variable to solve for
        num, den = sp.fraction(A_ij)
        
        # 3) Convert num, den to univariate polynomials in k_var
        num_poly = self._poly_in(k_var, num.as_expr())
        log.debug("Numerator of A_%d%d successfully polynomialized.", i, j)
        # Short-cicuit if den does not depend on k_var
        if not den.has(k_var):
            log.debug(f"A_{i}{j} does not have a {k_var}-dependent denominator, i.e. no additional poles in this matrix entry.")
            return num_poly, None
        den_poly = self._poly_in(k_var, den.as_expr())
        log.debug("Denominator of A_%d%d successfully polynomialized.", i, j)

        # 4) Gather metadata
        num_deg, num_even, num_free_params, num_u_sym, num_Q = self._gather_poly_metadata(num_poly, k_var)
        den_deg, den_even, den_free_params, den_u_sym, den_Q = self._gather_poly_metadata(den_poly, k_var)
        num_Q_poly = self._poly_in(k_var, num_Q.as_expr()) if num_Q is not None else num_Q
        den_Q_poly = self._poly_in(k_var, den_Q.as_expr()) if den_Q is not None else den_Q

        # 5) Return the dataclass
        num_poly_data = Poly(
            var=k_var,
            poly=num_poly,
            degree=num_deg,
            even=num_even,
            u=num_u_sym,
            u_poly=num_Q_poly,
            free_params=num_free_params,
        )
        den_poly_data = Poly(
            var=k_var,
            poly=den_poly,
            degree=den_deg,
            even=den_even,
            u=den_u_sym,
            u_poly=den_Q_poly,
            free_params=den_free_params,
        )

        # 6) Return matrix entry and denominator as polynomial in the Poly dataclass
        return  num_poly_data, den_poly_data
    
    def determinant_poly(self, solve_for: int = None, momentum : ArrayLike | None = None) -> Poly:
        '''
        Computed the determinant of G_inv and turns it into a polynomial.
        Returns the polynomial as a DetPoly object with its degree, evenness, reduced counterpart and so on
        according to the Poly dataclass.

        Parameters
        ----------
        solve_for: int or None
            identifies the variable, for which the polynomial is formed.
            If None, last dimension is chosen by default.
        momentum: ArrayLike or None
            Only relevant in numeric mode.
            Symbolic mode defaults to class inherent k_symbols.

        Returns
        -------
        Poly: dataclass
            Determinant as polynomial with metadata
        
        Raises
        ------
        ValueError
            If called in numeric mode.
        '''
        # 1) Build G^{-1}(k) and compute its determinant
        G_inv = self.greens_inverse(momentum)
        det = self._determinant(G_inv)
        if not self.symbolic:
            raise ValueError("The polynomial of the determinant cannot be constructed in numeric mode (symbolic = False). Returning its value for the given momentum.")
        solve_for = self._clean_solve_for(solve_for, dimension = self.d)
        k_var = self.k_symbols[solve_for]  # variable to solve for
        log.info("Constructing det(G_inv) as polynomial in variable %s.", k_var)
        
        # 2) Convert to a univariate polynomial in 'k_var'
        det_poly = self._poly_in(var = k_var, expr = det.as_expr())

        # 3) Gather metadata
        deg, even, free_params, u_sym, Q = self._gather_poly_metadata
        Q_poly = self._poly_in(k_var, Q.as_expr()) if Q is not None else Q

        # 5) Return the dataclass
        return Poly(
            var=k_var,
            poly=det_poly,
            degree=deg,
            even=even,
            u=u_sym,
            u_poly=Q_poly,
            free_params=free_params,
        )
    # endregion

    # region Poles
    def conditional_poles(self, include_adjugate: bool = True, include_determinant: bool = True, solve_for: int = None, case_assumptions: list = None) -> dict[str: ConditionSet]:
        '''
        Collects all poles contributing to the residue sum of the Fourier transform as Sympy ConditionSets so as to not freeze the program
        with heavy simplification of symbolic radicals. The caller can do this separately using these solution sets.

        Parameters
        -----------
        include_adjugate: bool or None
            Flag to include the zeros of the denominators of the entries of adj(G_inv)
            Defaults to True.
        include_determinant: bool or None
            Flag to include the zeros of det(G_inv)
            Defaults to True.

        Returns
        --------
        dict: {str: ConditionSet}
            str indicated the equation that causes the pole
            ConditionSet contains all the information to identify the symbolic pole

        Raises
        ------
        ValueError
            In numeric mode, i.e. if symbolic=False.
        '''
        log.debug("Starting conditional pole computation.")
        if not self.symbolic:
            raise ValueError("ConditionSets can only be computed in symbolic mode. Enable symbolic=True.")
        poles = {}
        if include_adjugate:
            A = self.adjugate_greens_inverse()
            rows, cols = A.shape
            for i in range(rows):
                for j in range(cols):
                    # Skip trivially zero entries early
                    if A[i, j] == 0:
                        continue
                    _, den_dc = self.numerator_denominator_poly(A,i,j,solve_for=solve_for)
                    if den_dc == None:
                        continue
                    pole_set = self._conditionset_for_Poly(den_dc)
                    poles[f"den(A_{i}{j})=0: ", pole_set]
                    log.debug(f"Entry A_{i}{j} has a pole.")
        if include_determinant:
            det_dc = self.determinant_poly(solve_for)
            solve_for = self._clean_solve_for(solve_for, self.d)
            k_var = self.k_symbols[solve_for]
            pole_set = self._conditionset_for_Poly(det_dc)
            poles[f"det(G_inv({k_var}))=0: ", pole_set]
            log.debug("Determinant poles successfully conditioned.")
        elif not include_adjugate:
            warnings.warn("No poles were included; returning empty dict. Enable include_adjugate or include_determinant.")
        log.info("Conditional pole computation complete.")
        return poles
    
    # After this point, all methods require numerical values for the parameters of polynomial
    # To that end, this little function provides the caller with all symbolic parameters which need to be evaluated:
    def required_parameters(self, expr: sp.Basic | Poly | sp.Poly | sp.Expr | None = None, solve_for: int = None) -> tuple[sp.Symbol,...]:
        """
        Get the set of SymPy symbols that need to be numerically evaluated.
        This includes all symbols present in the given expression aside from the variable to solve for: 'k_var'
        Caller can identify which symbols need to be defined for evaluation of the roots and the real space GF.
        Usually not needed for k-space GF evaluation.

        Returns
        -------
        Tuple of sp.Symbol object
            Symbols that must be defined for the Hamiltonian to be evaluated.
            Empty set in numeric mode.
        """
        if not self.symbolic:
            raise ValueError("There are no symbols in numeric mode. Enable symbolic = True.")
        expr = self.get_greens_inverse() if expr is None else expr
        if isinstance(expr, Poly):
            params = expr.free_params
        else:
            solve_for = self._clean_solve_for(solve_for, dimension = self.d)
            k_var = self.k_symbols[solve_for]
            params = expr.free_symbols - {k_var}
        return tuple(sorted(params, key=sp.default_sort_key))
    
    def det_poles(self,  vals: dict, solve_for: int = None, halfplane: str = None) -> dict[Union[float, sp.Basic]: Union[int, sp.Basic]]:
        log.debug("Starting exact pole computation for det(G_inv).")
        solve_for = self._clean_solve_for(solve_for, dimension=self.d)

        # 1) Unpacking poly dataclass
        det_poly_dc = self.determinant_poly(solve_for=solve_for) # dc: Poly dataclass
        k_var = det_poly_dc.var  # variable to solve for
        P = det_poly_dc.poly
        deg = det_poly_dc.degree
        params = det_poly_dc.free_params
        even = det_poly_dc.even
        log.info("Computing roots of det(G⁻¹(k))=0 for variable %s.", k_var)
        log.debug("det(G⁻¹(k)) is a %dth order polynomial in %s.", deg, k_var)
        # Short-circuit for constant:
        if not P.has(k_var):
            # constant in the solve variable -> either identically zero (singular) or no roots
            if sp.cancel(P).equals(0):
                raise ValueError(f"det G⁻¹ is identically zero; G⁻¹ is singular for all {k_var}.")
            warnings.warn(f"det(G⁻¹) is constant and non-zero in {k_var}; no roots to solve for, returning empty dict.")
            return {}
        
        # 2) Substituting input values
        P_eval = P.subs(vals)
        still_free = P_eval.free_symbols
        leftover = still_free - {k_var}
        required = set(params)
        if leftover & required:
             raise ValueError(
                 "Insufficient substsitution values provided in 'vals'"
                 f"det(G⁻¹) still contains unresolved symbols: {leftover & required}.", 
                 stacklevel=2)
        
        # 3) Root solving
        ### First, try root solving for the reduced polynomial:
        if even:
            Q = det_poly_dc.u_poly
            u = det_poly_dc.u
            assert Q.has(u), "Q should depend on variable u."
            Q_eval = Q.subs(vals)
            leftover_Q = Q_eval.free_symbols - {u}
            assert not leftover_Q, f"Reduced polynomial should have no unknown parameters after eval. Found: {leftover_Q}"
            
            
        return {}
    
    def denom_poles(self,  i: int, j: int, vals: dict, solve_for: int = None, halfplane: str = None, case_assumptions: list = None) -> list[tuple[str, sp.Set]]:
        return []
    
    def compute_roots_greens_inverse(self, solve_for: int = None, vals: dict = None, case_assumptions: list = None) -> tuple[dict, sp.Set]:
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

        solve_for = self._clean_solve_for(solve_for=solve_for, dimension=self.d)

        k_var = self.k_symbols[solve_for]  # variable to solve for
        log.info("Computing roots of G⁻¹(k) for variable %s.", k_var)

        # Define symbolic momentum components
        k = sp.Matrix(self.k_symbols)  # e.g., Matrix([k_x, k_y, k_z]) or fewer

        with sp.assuming(*predicates):
            # Compute the determinant of G_inv(k)
            G_inv = self.get_greens_inverse(k)
            log.debug("G⁻¹(k) constructed for root solving.")
            det_G_inv = self._determinant(G_inv)
            log.debug("det(G⁻¹(k)) successfully computed.")

            if not det_G_inv.has(k_var):
                # constant in the solve variable -> either identically zero (singular) or no roots
                if sp.simplify(det_G_inv).equals(0):
                    raise ValueError(f"det G⁻¹ is identically zero; G⁻¹ is singular for all {k_var}.")
                warnings.warn(f"det(G⁻¹) is constant and non-zero in {k_var}; no roots to solve for, returning empty set.")
                return [("det(G^{-1})=0", sp.EmptySet)]
            
            if vals is not None:
                det_G_inv = det_G_inv.subs(vals)
                log.debug("Substituted given numeric values into det(G⁻¹): %s", vals)
                free_symbols = det_G_inv.free_symbols
                free_symbols_list = list(free_symbols) # ordered list of free symbols to be returned with the result
                leftover_symbols = free_symbols - set(self.k_symbols)
                REQUIRED_SYMBOLS = self.get_required_symbols()
                if leftover_symbols & REQUIRED_SYMBOLS:
                    warnings.warn(
                        f"Insufficient substsitution values provided, det(G⁻¹) still contains unresolved symbols: {leftover_symbols & REQUIRED_SYMBOLS}. This may lead to failure or freezing during solving.", 
                        stacklevel=2)
            else:
                warnings.warn("No numeric substitutions values provided; solving symbolically with parameters may freeze or fail.", stacklevel=2)
                free_symbols_list = list(det_G_inv.free_symbols)
            try:
                # 1) Set up as polynomial in k_var
                log.debug("Attempting polynomial root solving in %s.", k_var)
                poly = sp.Poly(det_G_inv, k_var, domain=sp.EX)  # EX is robust with symbols/parameters
            except PolynomialError:
                # 2) Short-circuit for non-polynomial determinants
                raise TypeError(f"Expected det(G_inv) to be a polynomial in {k_var}. Solver cannot handle other types.")
            else: 
                # Continue wth 1), since no PolynomialError was raised.
                if poly.total_degree() <= 0:
                    warnings.warn(f"det(G⁻¹) polynomial degree is zero in {k_var}, but det_G_inv.has(k_var) is {det_G_inv.has(k_var)}. Earler warning should have caught this.")
                    return {}, free_symbols_list
                deg = sp.degree(poly, k_var)
                log.debug("det(G⁻¹) is polynomial of degree %d in %s.", deg, k_var)
                reduced = self._try_even_reduction(poly, k_var)
                # a) Even-power reduction case:
                if reduced is not None:
                    reduced_poly, k_squared = reduced
                    log.debug("Polynomial reduction successful: substituting %s²=%s. Solving for roots in new variable", k_var, k_squared)
                    # cubic (usually): solve exactly & quickly
                    reduced_poly = reduced_poly.as_poly(k_squared)
                    try:
                        t_roots = func_timeout(TIMEOUT_GATE, self._factorize_poly_and_find_roots, args = (reduced_poly, k_squared))
                    except FunctionTimedOut:
                        raise RuntimeError(f"Sympy factoring and root solving exceeded {TIMEOUT_GATE} seconds.")
                    k_roots = {}
                    for ti, m in t_roots.items():
                        # branch lift: ±sqrt(t_i)
                        k_roots[sp.sqrt(ti)] = m
                        k_roots[-sp.sqrt(ti)] = m
                        #CAREFUL! IF ti==0, then both are the same and multiplicity needs to be doubled
                    log.info("All roots of det(G⁻¹)=0 successfully computed with even-power reduction.")
                    log.debug("There are %d unique roots.", len(k_roots))
                    return k_roots, free_symbols_list
                # b) General polynomial case
                log.debug("No even-power reduction possible; solving as general polynomial.")
                try:
                    k_roots = func_timeout(TIMEOUT_GATE, self._factorize_poly_and_find_roots, args = (poly, k_var))
                except FunctionTimedOut:
                    raise RuntimeError(f"Sympy factoring and root solving exceeded {TIMEOUT_GATE} seconds.")
                log.info("All roots of det(G⁻¹)=0 successfully computed polynomially.")
                log.debug("There are %d unique roots.", len(k_roots))
                return k_roots, free_symbols_list
    # endregion

    # region Real-space
    # -- Symbolic 1D real-space Fourier transform --
    def compute_rspace_greens_symbolic_1d_along_last_dim(self,
                                                         z: float | sp.Basic,
                                                         z_prime: float | sp.Basic,
                                                         z_diff_sign: int = None,
                                                         full_matrix: bool = False,
                                                         vals: dict = None,
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

        #for i, lambda_i in enumerate(eigenvalues):
        #    contrib, contributed_any = self._residue_sum_for_lambda(
        #        i, lambda_i, z, z_prime, k_dir, z_diff_sign, predicates, choices)
        #    has_contributions = has_contributions or contributed_any
        #    assert contrib == 0 if not contributed_any else True, "If no poles contributed, the contribution must be zero."
        #    G_z_diag.append(contrib)

        #    log.debug("Processed eigenvalue λ_%d=%s.", i, lambda_i)
        #    log.debug("Diagonal entry to G(z,z') #%d: %s", i, contrib)
        self._finalize_ambiguities_or_raise(context="root solving")   

        #if not has_contributions:
        #    warnings.warn(
        #        "No poles passed the sign check; returning zero Green's function.", stacklevel=2)
        #    assert sp.diag(*G_z_diag) == sp.zeros(self.N), "Expected zeros on the diagonal if no poles contributed."

        #elif all((val.is_zero is True) for val in G_z_diag):
        #    warnings.warn(
        #        "Green's function is identically zero: all residue contributions canceled out.", stacklevel=2)
        #    assert has_contributions == True, "Expected has_contributions=True if some poles contributed, otherwise earlier warning should have been triggered."

        G_z = sp.diag(*G_z_diag)
        log.info("1D real-space %s Green's function G(%s,%s) successfully computed.", self.green_type, z, z_prime)
        if assumptions_made == True:
            log.info(f"Note: Additional assumptions were made. Keep in mind.") 
            log.debug("Case assumptions: %s", case_assumptions)

        # Note: Currently returning only diagonal Green's function G(z, z′)
        # Full matrix reconstruction from eigenbasis can be added if needed:
        if full_matrix:
            #G_full = eigenbasis @ G_z @ invert_matrix(
            #    eigenbasis, symbolic=True)
            log.info("The full Green's function matrix in the original basis is returned.")
            #return G_full
            
        log.info("Note: Only diagonal entries are returned by default. Choose full_matrix=True for the full matrix in the original basis.") 
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
    
    @staticmethod
    def _clean_solve_for(solve_for: int | None, dimension: int):
        d = dimension
        if solve_for is None:
            solve_for = d - 1  # default to last dimension
            log.debug("No solve_for input provided. Defaulting to solve_for=%d", solve_for)
        # validate index
        if not isinstance(solve_for, int):
            raise TypeError(
                f"'solve_for' must be an int in [0, {d-1}] (k-dimension index).")
        if solve_for < 0 or solve_for >= d:
            valid_indices = ", ".join(str(i) for i in range(d))
            raise ValueError(
                f"'solve_for' out of range: got {solve_for}, valid indices are {{{valid_indices}}}.")
        
    @staticmethod
    def _poly_in(var: sp.Symbol, expr: sp.Expr, *, zero_equiv: bool = True, domain=sp.EX) -> sp.Poly:
        """
        Build a *univariate* Poly in `var` from a SymPy expression `expr`.

        - Uses together() to collect fractions, cancels common factors in `var`,
        and clears any *var-dependent* denominators.
        - If `zero_equiv=True` (default), when a var-dependent denominator remains,
        it returns the *numerator after cancellation* so the returned polynomial
        has the same zero set as `expr` (ignoring poles). This is what you want
        for root-finding of det(G^{-1})=0.
        - Raises TypeError if `expr` is not (rational) polynomial in `var` 
        (e.g., contains sqrt(var), exp(var), Abs(var), non-integer powers).

        Parameters
        ----------
        var : sympy.Symbol
            The solve-for variable (e.g., k_z).
        expr : sympy.Expr
            The expression to convert (e.g., det(G^{-1})).
        zero_equiv : bool, optional
            If True, clear var-dependent denominators by returning the cancelled
            numerator so zeros are preserved. If False, require a true polynomial.
        domain : sympy domain, optional
            Poly coefficient domain. Use sp.EX to allow parameter-rational coeffs.

        Returns
        -------
        sp.Poly
            Univariate polynomial in `var`.

        Examples
        --------
        >>> P = poly_in(kz, detGinv_expr)          # OK if det/den has kz
        >>> P.degree(), P.gens
        >>> # If you need a strict polynomial (no denominator clearing):
        >>> P = poly_in(kz, detGinv_expr, zero_equiv=False)
        """
        # Normalize rational structure and cancel common factors
        expr = sp.cancel(expr)
        num, den = sp.fraction(expr) 

        # Quick path: try a true polynomial as-is
        try:
            P = sp.Poly(sp.expand(expr), var, domain=domain)
            return P
        except sp.PolynomialError:
            pass  # fall through and attempt clearing variable-dependent denominators

        # If denominator has `var`, clear it in a zero-equivalent sense
        if den.has(var):
            # Ensure numerator is polynomial in var
            try:
                numP = sp.Poly(sp.expand(num), var, domain=domain)
            except PolynomialError as _:
                raise ValueError(f"Denomenator of expr depends on {var} and numerator is not polynomial in {var}. Investigate.")
            # Also try to ensure denominator *as poly in var* to strip any residual gcd
            try:
                denP = sp.Poly(sp.expand(den), var, domain=domain)
                log.debug("Numerator and denominator are both polynomials in %s.", var)
            except sp.PolynomialError:
                denP = None
                log.debug("Numerator is polynomial in %s; denominator depends non-polynomially on %s.", var, var)

            # Cancel any remaining common polynomial factor in var
            if denP is not None:
                #cancelling greatest common divisor, in case sp.cancel missed something
                g = sp.gcd(numP, denP)
                if g.degree() > 0:
                    numP = numP.quo(g)
                    #denP = denP.quo(g)

            if zero_equiv:
                return sp.Poly(numP.as_expr(), var, domain=domain)
            else:
                raise TypeError(
                    f"Expression is not a true polynomial in {var} "
                    "(var-dependent denominator present). Set zero_equiv=True to clear it safely for root-finding."
                )

        # Denominator does not depend on var → coefficients may be rational in params
        try:
            return sp.Poly(sp.expand(num/den), var, domain=domain)
        except sp.PolynomialError:
            raise
    
    @staticmethod
    def _gather_poly_metadata(poly: sp.Poly, var: sp.Symbol):
        def all_even_powers(P: sp.Poly) -> bool:
            return all(e % 2 == 0 for (e,), _ in P.terms())
        deg = poly.degree()
        even = all_even_powers(poly)
        free_params = tuple(sorted(poly.free_symbols - {var}, key=lambda s: s.sort_key()))

        # 4) If even in k_var, build reduced poly Q(u) with u=k_var^2
        u_sym = None
        Q = None
        if even and deg > 0:
            u_sym = sp.Symbol(f"{str(var)}²", real=True)
            # Rebuild Q(u) from det_poly(k_var) terms:
            Q_terms = []
            for (e,), c in poly.terms():
                # e is guaranteed even here
                Q_terms.append(c * u_sym**(e // 2)) # exponent halved
            Q = sp.add(*Q_terms) if Q_terms else 0
            Q = sp.Poly(Q.as_expr(), u_sym, domain=sp.EX) # turn into sp.Poly object
        else:
            u_sym = None
            Q = None
        return deg, even, free_params, u_sym, Q

    @staticmethod    
    def _conditionset_for_Poly(poly_dataclass: Poly) -> sp.Set:
        """
        Build a ConditionSet for the polynomial == 0 over Complexes.
        poly_dc is your Poly dataclass (with fields: var, poly, degree, even, u, u_poly, free_params).
        Handles edge cases: identically 0 or constant != 0.
        """
        x = poly_dataclass.var
        expr = poly_dataclass.poly.as_expr()

        # Quick dependency check
        if not expr.has(x):
            # Constant case: either all Complexes (identically zero) or empty
            if expr.equals(0):
                warnings.warn("Expression is identically zero. Returning sp.Complexes instead of ConditionSet.")
                return sp.Complexes   # All complex numbers satisfy 0 == 0
            else:
                warnings.warn(f"Expression is constant in {x} and non-zero. Returning EmptySet instead of ConditionSet.")
                return sp.EmptySet    # No solution to c == 0 with c ≠ 0

        # General case: zero set of expr in Complexes
        return sp.ConditionSet(x, sp.Eq(expr, 0), sp.Complexes)

    def _compile_polynomials(self, poly: Poly | sp.Poly | sp.Expr, *, var: sp.Symbol = None, backend: str = "mpmath", prec: int = 80):
        """
        Compile P(k; params), P'(k; params) and optionally a matrix Nij(k; params)
        into fast numeric callables with a deterministic parameter order.

        Parameters
        ----------
        P : ExprLike
            Main polynomial/expression in the solve-for variable (e.g., k_z).
        var : sp.Symbol, optional
            Solve-for variable. If None, inferred (deterministically) from symbols.
        params : Iterable[sp.Symbol], optional
            Fixed parameter order. If None, inferred from free symbols minus 'var'.
        Nij : MatrixLike, optional
            A matrix (list of lists or sp.Matrix) of expressions/polynomials to compile.
        backend : {"numpy", "mpmath"}
            Backend fed into sympy.lambdify.
        prec : int
            Precision hint for mpmath workflows (set mp.mp.dps externally when using).
        use_even_speedup : bool
            If True, tries to factor P(k) into Q(u) with u=k**2 when P is even in k and
            uses P'(k) = 2*k*Q'(k**2). Falls back to plain d/dk if unsuccessful.

        Returns
        -------
        GenericCompiled
            Holder with callables and metadata.
        """
        if backend not in ("numpy", "mpmath"):
            raise ValueError("backend must be 'numpy' or 'mpmath'")
        modules = backend # used for sp.lambdify

        if isinstance(poly, Poly):
            poly_dc = poly # differentiate dataclass Poly from sp.Poly object
            k = poly_dc.var
            P = poly_dc.poly # sp.Poly object now
            deg = poly_dc.degree
            even = poly_dc.even
            params = poly_dc.free_params
            if even:
                u = poly_dc.u # squared variable
                u_poly = poly_dc.u_poly # reduced polynomial for u, sp.Poly object
        elif var is None:
            raise ValueError("No variable provided. If poly is not a Poly dataclass, 'var' needs to be provided.")
        elif isinstance(poly, Union[sp.Expr, sp.Basic]):
            poly = poly.as_expr()
            poly = self._poly_in(var, poly) # sp.Poly object now
        if isinstance(poly,sp.Poly):
            k = var
            deg, even, params, u, u_poly = self._gather_poly_metadata(poly, k)
            P = poly # consistent naming between cases

        # Try even reduction
        if even:
            assert u is not None, "for an even polynomial, a squared variable should have been generated."
            Q = u_poly
            assert Q.has(u), "Q should depend on variable u."
            dQ_du = Q.diff(u) # still sp.Poly 
            dP_dk = sp.cancel(2 * k * dQ_du.subs({u: k**2}))
        else:
            dQ_du = None
            dP_dk = P.diff(k)

        # Lambdify P and P'
        P = sp.lambdify((k, *params), P, modules=modules)
        dP_dk = sp.lambdify((k, *params), dP_dk, modules=modules)
        if Q is not None:
            Q = sp.lambdify((u, *params), Q, modules=modules)
            dQ_du = sp.lambdify((u, *params), dQ_du, modules=modules)

        return PolyCompiled(
            var=k,
            params=params,
            backend=backend,
            prec=prec,
            even=even,
            P=P,
            Pprime=dP_dk,
            u=u,
            Q=Q,
            Qprime=dQ_du,
        )

    def _determinant(self, matrix: MatrixLike) -> sp.Basic | complex:
        """
        Compute the determinant of a matrix, symbolic or numeric.

        Parameters
        ----------
        matrix : MatrixLike
            The input matrix (sympy.Matrix or np.ndarray).

        Returns
        -------
        det : sympy.Basic or complex
            The determinant of the matrix.
        """
        A = sanitize_matrix(matrix, symbolic=self.symbolic, expected_size=self.N)
        if not self.symbolic:
            det_A = np.linalg.slogdet(A)
            log.debug("Determinant computed numerically with NumPy.")
            return det_A
        det_A = A.berkowitz_det()
        log.debug("Determinant computed symbolically with Berkowitz algorithm")
        return det_A
    
    def _try_even_reduction(self, det, k):
        # det is det(G^{-1})(k); k is the solve variable
        # Check evenness: D(-k) == D(k)
        if sp.simplify(sp.together(det.subs(k, -k) - det)) != 0:
            return None  # not even

        # Build P(t) with t = k**2
        poly_k = sp.Poly(sp.together(det), k, domain=sp.EX)
        coeffs = {}
        for (exp,), c in poly_k.terms(): # (e,c) pairs: exponent tuple (with one entry), coefficient
            if exp % 2 != 0:
                return None  # odd power sneaked in; bail out
            coeffs[exp // 2] = coeffs.get(exp // 2, 0) + c
        t = sp.Symbol("t", complex=True)  # auxilliary variable; t = k**2
        poly_t = sp.Poly(sum(c * t**e for e, c in coeffs.items()), t, domain=sp.EX)
        return poly_t, t
    
    def _poly_roots(self, poly: sp.Poly, var) -> dict:
        # Garantee univariate polynomial:
        poly = sp.Poly(poly, var, domain="EX")
        if poly.degree() <= 0:
            warnings.warn(f"Poly constant in {var}. Should have triggered earlier. Returning empty dict.")
            return {}
        # First attempt: exact root solving directly with poly.roots()
        try:
            log.debug("Attempting exact root solving with poly.roots().")
            roots = func_timeout(TIMEOUT_GATE, poly.roots)
            log.debug("Direct root solving successful.")
            return roots
        except FunctionTimedOut:
            warnings.warn(f"Exact poly.roots() solving exceeded {TIMEOUT_GATE} seconds.")
        # Second attempt: exact root solving indirectly via factorization
        try: 
            log.debug("Attempting root solving with factorization.")
            # Square-free decomposition:
            _, factors = func_timeout(TIMEOUT_GATE,poly.sqf_list) # factors is a list of (factor, multplicity) tuples
            log.debug("Square-free decomposition of poly successful.")
            roots = {}
            for f, mult in factors:
                f_roots = func_timeout(TIMEOUT_GATE,f.roots)
                for r in f_roots.keys():
                    m = f_roots[r]*mult # update multiplicity with decomposition factor
                    roots[r] = roots.get(r, 0) + m
            log.debug("Root solving via factorization successful.")    
            return roots        
        except FunctionTimedOut:
            warnings.warn(f"Root solving with factorization exceeded {TIMEOUT_GATE} seconds.")
        # Third attempt: numerical root solving with poly.nroots()
        try:
            log.debug("Attempting numerical root solving with poly.nroots().")
            # Numerical root approximation before clustering:
            roots_list = func_timeout(TIMEOUT_GATE,poly.nroots) # returns list of roots as float objects
            log.debug("Roots successfully approximated numerically; not yet clustered.")
            # Sort deterministically by (Re, Im)
            roots_sorted = sorted(roots_list, key=lambda z: (sp.re(z), sp.im(z)))
            # Clustering effectively equal roots
            clusters = []  # list of 2-item lists[[mean_value, mult], ...]
            for z in roots_sorted:
                if clusters and abs(z - clusters[-1][0]) <= NUM_TOL: # due to ordering, checking last added is sufficient 
                    # weighted centroid update for stability
                    c, m = clusters[-1]
                    clusters[-1] = [(c*m + z) / (m + 1), m + 1] # [new mean, new mult)]
                else:
                    clusters.append([z, 1])
            # Canonicalize keys: round to 'DIGITS' to avoid tiny noise
            roots = {}
            for c, m in clusters:
                c_key = sp.N(c, digits=DIGITS) # SymPy Float/ComplexFloat at given precision
                roots[c_key] = roots.get(c_key, 0) + m
            log.debug("Clustering of root approximations successfull. Roots dict returned.")
            return roots
        except FunctionTimedOut:
            raise RuntimeError(f"All attempts at root solving exceeded {TIMEOUT_GATE} seconds. Investigate.")

    def _factorize_poly_and_find_roots(self, poly, variable):
        constants, factors = sp.factor_list(poly.as_expr())
        log.debug("Factorization of polynomial successful. There are %d factors.", len(factors))
        roots = {}
        for i,(r,_) in enumerate(factors):
            log.debug("Degree of factor #%d is %d.", i, sp.degree(r, variable))
            roots_i = sp.roots(r.as_expr(), variable)  # dict {t_i: mult}
            log.debug("Roots of reduced factor #%d were successfully computed.", i)
            roots.update(roots_i)
        return roots

    def _im_sign_of_root(self, k0, i, n, predicates=None, choices=None):
        """
        Try to determine sign(Im(k0)) robustly.
        Returns +1, -1, or None if undecidable.
        """
        predicates = predicates or []
        choices = choices or {}
        log.debug("Determining sign of Im(k0) for root k0=%s", k0)

        # 1) Direct attempt under assumptions
        with sp.assuming(*predicates):
            s = sp.sign(sp.im(k0))
            if s in (sp.Integer(1), sp.Integer(-1)):
                return int(s)
            if s == 0:
                return 0
            log.debug("Direct sign(Im(k0)) check inconclusive.")

        # 2) c*sqrt(a + I*b) pattern
        ## if c is positive, the sign of Im(k0) is the sign of b
        base = sp.cancel(k0)
        sign_flip = 1
        if base.func is sp.Mul:
            # factor out explicit -1 if present
            coeffs = [c for c in base.args if c.is_number]
            coefficient = sp.Mul.fromiter(coeffs) if coeffs else sp.Integer(1)
            log.debug("Full coefficient factored out: %s", coefficient)
            negs = [c for c in coeffs if c < 0]
            n = len(negs)
            log.debug("%d negative coefficients", n)
            sign_flip = (-1)**n
            assert sp.sign(coefficient) == sign_flip, "Coefficient sign mismatch."
            # separate sqrt factors from others
            base = sp.expand(base/coefficient)
        if base.func is sp.sqrt:
            log.debug("k0 is a sqrt function.")
            base_squared = sp.simplify(sp.cancel(k0**2))
            log.debug("k0^2 = %s ", base_squared)
            with sp.assuming(*predicates):
                b = sp.im(base_squared)
                log.debug("Imaginary part of k0^2: Im(k0^2) = %s", b)
                s = sp.sign(sp.simplify(b))
                if s in (sp.Integer(1), sp.Integer(-1)):
                    return sign_flip * int(s)
                if s == 0:
                    return 0
                log.debug("Sign of Im(k0^2) inconclusive.")
        else:
            choice_key = ("im_sign_root", f"lambda_{i}.root_{n}.sqrt_form")
            if choices.get(choice_key) is True:
                log.debug(f"{choice_key[1]} ambiguity resolved by choice provided.")
                log.debug("Assuming k0 is a sqrt function.")
                base_squared = sp.simplify(sp.cancel(k0**2))
                log.debug("k0^2 = %s ", base_squared)
                with sp.assuming(*predicates):
                    b = sp.im(base_squared)
                    log.debug("Imaginary part of k0^2: Im(k0^2) = %s", b)
                    s = sp.sign(sp.simplify(b))
                    if s in (sp.Integer(1), sp.Integer(-1)):
                        return sign_flip * int(s)
                    if s == 0:
                        return 0
                    log.debug("Sign of Im(k0^2) inconclusive.")
            elif choices.get(choice_key) is False:
                log.debug(f"{choice_key[1]} ambiguity resolved by choice provided.")
                log.debug("Assuming k0 is NOT a sqrt function.")
            else:
                self._add_amb(
                    where="im_sign_root",
                    what=f"lambda_{i}.root_{n}.sqrt_form",
                    predicate=None,  
                    options=[True, False],
                    consequence="Cannot decide, if k0 is a sqrt function.",
                    data={"k0": k0, "base without coeffs": base, "base func": base.func},
                    severity="error"  # unresolved unless a choice or predicate resolves it
                )
                return s

        # 3) η -> 0+ limit (if an eta symbol is known/used)
        ## Taylor expansion around η=0 up to second order
        log.debug("Trying Taylor expansion to determine sign(Im(k0)).")
        assert k0.has(self.eta), "Expected k0 to depend on eta for limit test."
        with sp.assuming(*predicates):
            try:
                zero_order = k0.subs(self.eta, 0)
                first_order = sp.diff(k0, self.eta).subs(self.eta, 0)
                second_order = sp.diff(k0, self.eta, 2).subs(self.eta, 0)
                if sp.im(zero_order) != 0:
                    log.debug("Im(k0) at η=0 is non-zero.")
                    s = sp.sign(sp.im(zero_order))
                    if s in (sp.Integer(1), sp.Integer(-1)):
                        return int(s)
                    if s == 0:
                        return 0
                    log.debug("Sign of Im(k0) at η=0 inconclusive.")
                elif first_order != 0:
                    log.debug("Im(k0) at η=0 is zero, Im(d k0/d eta) is not.")
                    s = sp.sign(sp.im(first_order))
                    if s in (sp.Integer(1), sp.Integer(-1)):
                        return int(s)
                    if s == 0:
                        return 0
                    log.debug("Sign of first order expansion Im(k0) at η=0 inconclusive.")
                elif second_order != 0:
                    log.debug("First order expansion Im(k0) at η=0 is zero, Im(d^2 k0/d eta^2) is not.")
                    s = sp.sign(sp.im(second_order))
                    if s in (sp.Integer(1), sp.Integer(-1)):
                        return int(s)
                    if s == 0:
                        return 0
                    log.debug("Sign of second order expansion Im(k0) at η=0 inconclusive.")
                    pass
                else:
                    log.debug("Second order expansion Im(k0) at η=0 is zero.")
                    log.debug("Taylor expansion inconclusive.")
            except Exception as e:
                log.error("Taylor expansion failed: %s", e)
                pass
        # 4) Final fallback: manual disambiguation via choice
        choice_key = ("im_sign_root", f"lambda_{i}.root_{n}.im_sign")
        if choices.get(choice_key) in (+1, -1, 0):
            log.debug(f"{choice_key[1]} ambiguity resolved by choice provided.")
            return choices[choice_key]
        else:
            self._add_amb(
                where="im_sign_root",
                what=f"lambda_{i}.root_{n}.im_sign",
                predicate=None,  
                options=[+1, -1, 0],
                consequence="Cannot decide sign(Im(k0)).",
                data={"k0": k0, "Im(k0)": sp.im(k0), "attempted methods": ["direct", "sqrt_pattern", "taylor_eta"]},
                severity="error"  # unresolved unless a choice or predicate resolves it
            )
            return s
   
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
            if offdiag_norm > NUM_TOL:
                raise ValueError(
                    f"Eigendecomposition inconsistent: off-diagonal norm {offdiag_norm:.2e} exceeds {NUM_TOL:.1e}"
                )
            # (2) Diagonal entries should match eigenvalues within tolerance (order already enforced by sorting)
            diag_diff = np.diag(D_recon) - np.diag(D)
            # normalization for appropriate tolerance scaling:
            rel_err = np.linalg.norm(diag_diff) / \
                max(1.0, np.linalg.norm(np.diag(D)))
            if rel_err > NUM_TOL:
                raise ValueError(
                    f"Eigendecomposition inconsistent: diagonal mismatch rel. error {rel_err:.2e} exceeds {NUM_TOL:.1e}"
                )
            # (3) Final reconstruction check
            # Frobenius norm ought to vanish
            residual_norm = np.linalg.norm(P @ D @ P_inv - G_inv)
            # normalization for appropriate tolerance scaling
            den = max(1.0, np.linalg.norm(G_inv))
            if residual_norm / den > NUM_TOL:
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

            # factorize lambda_i (polynomial with known roots):
            leading_coeff = sp.Poly(lambda_i, kz_sym).LC() 
            factors = [(kz_sym - r)**m for r, m in roots_with_mult.items()]
            lambda_fact = leading_coeff * sp.Mul.fromiter(factors)
            log.debug("lambda_fact = %s", lambda_fact)
            zero_check = sp.simplify(lambda_fact - lambda_i)
            log.debug("Must be zero: lambda_fact - lambda_i = %s", zero_check)
            assert zero_check.is_zero or zero_check.equals(0), "Factorization must not change lambda."

            residue_sum = 0
            for n, (k0, m) in enumerate(roots_with_mult.items()):  # roots k0 with their multiplicity m
                log.debug("root_%d of lambda_%d: k0 = %s, m = %s", n, i, k0, m)
                assert kz_sym not in k0.free_symbols, "root must not depend on kz_sym" # safeguard

                # Halfplane selection
                sgn = self._im_sign_of_root(k0, i, n, predicates=predicates, choices=choices)
                if sgn not in (-1, 0, 1):
                    log.debug("Sign of Im(k0) for root k0 = %s of lambda_%d could not be determined.", k0, i)
                    continue  # ambiguity recorded; skip this root for now
                elif sgn in (-1, 1) and int(sgn) != z_diff_sign:
                    log.debug("Pole %s=%s (m=%s) of lambda_%d lies in wrong half-plane; skipped.", kz_sym, k0, m, i)
                    continue
                elif sgn == 0:
                    raise ValueError(
                        f"Pole at {kz_sym}={k0} (m={m}) lies on the real axis; integral is ill-defined. Provide finite broadening η.")
                assert sgn == z_diff_sign, "If we reach this point after halfplane selection, the signs must match."

                # Residue formula for pole of order m:
                # Res = 1/(m-1)! * d^{m-1}/dk^{m-1} [ (k-k0)^m * phi / lambda_i(k) ] at k=k0
                fraction = sp.cancel((kz_sym - k0)**m *phase/ lambda_fact)
                log.debug("residue fraction = %s", fraction)
                log.debug("free symbols in fraction: %s", fraction.free_symbols)
                if m == 1:
                    res = sp.simplify(fraction.subs(kz_sym, k0))  # zero-th derivative
                    log.debug("after subs and simpl: res = %s", res)
                else:
                    deriv = sp.cancel(sp.diff(fraction, (kz_sym, m - 1)))
                    res = sp.simplify(sp.cancel(deriv.subs(kz_sym, k0) / sp.factorial(m - 1)))
            
                residue_sum += res
                contributed_any = True
                log.debug("Pole %s=%s (m=%s) contributed to the residue sum.", kz_sym, k0, m)
                log.debug("res contrib: %s", res)
                log.debug("res sum snapshot: %s", residue_sum)

            contrib = sp.I * residue_sum  # factor of i from residue theorem
            log.debug("lambda_%s: %s ", i, contrib)

        return contrib, contributed_any
    # endregion
