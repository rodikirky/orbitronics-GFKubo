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
import mpmath as mp
mp.mp.dps = 50 # mp precision
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
    label: str
    degree: int                    # degree in `var`
    even: bool                     # True if all exponents of `var` are even
    u: Optional[sp.Symbol] = None  # if even: u = var**2
    u_poly: Optional[sp.Poly] = None  # if even: Q(u) with P(var)=Q(var**2)
    free_params: Tuple[sp.Symbol, ...] = ()  # symbols in P other than `var`

@dataclass(frozen=True)
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
    #even: bool
    P: callable
    #Pprime: callable
    #u: Optional[sp.Symbol] = None
    #Q: Optional[callable] = None
    #Qprime: Optional[callable] = None

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
        # Hamiltonian:
        self.H = hamiltonian
        if not callable(self.H):
            raise ValueError(
                "Hamiltonian must be a callable function H(k).")
        
        # Choice of dimension determines default momentum symbols:
        self.d = int(dimension)
        if self.d not in (1, 2, 3):
            raise ValueError(
                f"Only 1D, 2D, and 3D systems are supported. Got dimension={self.d}.")
        
        # Canonical momentum symbols used internally for solving:
        # k_symbols[0] = "k_x", k_symbols[1] = "k_y", k_symbols[2] = "k_z"
        names = ["k"] if self.d == 1 else [
            f"k_{ax}" for ax in "xyz"[:self.d]]
        # not limited to real numbers since complex values must be allowed for integration
        self.k_symbols = sp.symbols(" ".join(names))
        # For consistency in code paths, make it indexable like a list
        if isinstance(self.k_symbols, sp.Symbol):
            self.k_symbols = [self.k_symbols]
        assert isinstance(self.k_symbols, (list, tuple)
                            ) and len(self.k_symbols) == self.d
        self.k_vec = sp.Matrix(self.k_symbols)
        k_for_H = (self.k_vec[0] if self.d == 1 else self.k_vec) # H(k) expects scalar if d=1
        self.H_k = sp.Matrix([self.H(k_for_H)]) if self.d==1 else sp.Matrix(self.H(k_for_H))
        if not (hasattr(self.H_k, "shape") and self.H_k.shape[0] == self.H_k.shape[1]):
            raise ValueError(f"Hamiltonian must return a square matrix.")
        
        # band size, e.g., 2 for spin-1/2 systems
        self.N = int(self.H_k.shape[0])
        self.identity = sp.eye(self.N)

        # Symbolic parameters:    
        self.omega = sp.symbols("omega", real=True) # energy level
        self.eta = sp.symbols("eta", real=True, positive=True) # broadening

        # Green's function type
        self.q = 1 if retarded else -1
        self.green_type = "retarded (+iη)" if self.q == 1 else "advanced (−iη)"

        #log.debug("Initialized %r", self) # developer snapshot for logs
        log.info("Initialized %s", self) # readable banner for operators/notebooks
        self._ledger = AmbiguityLedger()
    
    def __repr__(self):
        try:
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
                    f"N={self.N}, d={self.d}, "
                    f"ω={self.omega}, η={self.eta}, type={self.green_type}, "
                    f"I={I_summary}, H={H_name}, k={k_summary})")
        except Exception:
            return f"{self.__class__.__name__}(unprintable; id=0x{id(self):x})"

    def __str__(self):
        try:
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
                f"  N×N: {self.N}×{self.N}   d: {self.d}\n"
                f"  ω: {self.omega}   η: {self.eta}   type: {self.green_type}\n"
                f"  identity: {identity_summary}"
                f"{k_line}"
            )
        except Exception:
            return f"{self.__class__.__name__} (unprintable)"
    # endregion

    # region k-space
    def greens_inverse(self, momentum: ArrayLike | None = None, vals: dict = None) -> MatrixLike:
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

        if momentum is None:
            momentum = sp.Matrix(self.k_symbols)
            H_k = self.H_k
        else:
            k_vec = sanitize_vector(momentum, symbolic=False, expected_dim=self.d) # ensure iterable and correct type and shape
            k_for_H = (k_vec[0] if self.d == 1 else k_vec) # scalar only for H(k), since H expects scalar, if d=1
            log.debug("Calling H(k) for %s; k=%s", getattr(self.H, "__name__", type(self.H).__name__), k_for_H)  
            H_k = sanitize_matrix(H_k, symbolic=True, expected_size=self.N)
            H_k = self.H(k_for_H)  # Hamiltonian at momentum k
        log.debug("Computing G^{-1}(k) with: momentum=%s", momentum)

        imaginary_unit = sp.I if self.symbolic else 1j
        G_inv = (self.omega + self.q * self.eta *
                 imaginary_unit) * self.I - H_k
        log.debug("Built G^{-1}(k) = (ω %s iη)I - H(k)", "+" if self.q==1 else "-")
        if vals is not None:
            G_inv_eval = G_inv.subs(vals)
            log.debug("Returning G^{-1}(k) with the values provided")
            log.debug(f"Free symbols remaining: {G_inv_eval.free_symbols}")
            return sp.simplify(G_inv_eval)
        return G_inv
    
    def kspace_greens_function(self, momentum: ArrayLike | None = None, vals: dict = None) -> MatrixLike:
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
        G_inv = self.greens_inverse(momentum)
        G_k = invert_matrix(G_inv, symbolic=True)
        log.info("G(k) computed successfully.") 
        log.debug("G(k): shape=%s, backend=%s", getattr(G_k,"shape",None), "sym" if self.symbolic else "num")
        if vals is not None:
            G_k_eval = G_k.subs(vals)
            log.debug("Returning G(k) evaluated at the values provided.")
            log.debug(f"Free symbols remaining: {G_k_eval.free_symbols}")
            return sp.simplify(G_k_eval)     
        return G_k
    
    def adjugate_greens_inverse(self, momentum: ArrayLike | None = None, vals: dict = None) -> MatrixLike:
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
        G_inv = self.greens_inverse(momentum)
        det = self._determinant(G_inv)
        adjugate = G_inv.adjugate() if self.symbolic else det*np.linalg.inv(G_inv)
        if vals is not None:
            adjugate_eval = adjugate.subs(vals)
            log.debug("Returning adjugate matrix evaluated at the values provided.")
            log.debug(f"Free symbols remaining: {adjugate_eval.free_symbols}")
            return adjugate_eval
        return adjugate
    
    def numerator_denominator_poly(self, ratio: sp.Basic, i: int, j: int, solve_for: int = None) -> tuple[Poly, Poly]: 
        '''
        Selects a matrix entry according to the indeces provided and checks whether that entry has a denominator that depend in the 
        variable to solve for 'k_var'.
        If so, this method returns the numerator and the denominator of the matrix element as polynomials in k_var
        in form of the Poly dataclass.

        Parameters
        ----------
        ratio: sp.Basic
            Rational function with polynomials in numerator and denominator.
            usually the i,j-entry of the adjugate of G_inv
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
        A_ij = sp.cancel(ratio)
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
            label=f"num(A_{i}{j}({k_var})",
            degree=num_deg,
            even=num_even,
            u=num_u_sym,
            u_poly=num_Q_poly,
            free_params=num_free_params
        )
        den_poly_data = Poly(
            var=k_var,
            poly=den_poly,
            label=f"denom(A_{i}{j}({k_var})",
            degree=den_deg,
            even=den_even,
            u=den_u_sym,
            u_poly=den_Q_poly,
            free_params=den_free_params
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
            label=f"det(G⁻¹({k_var}))",
            degree=deg,
            even=even,
            u=u_sym,
            u_poly=Q_poly,
            free_params=free_params
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
    
    def poly_poles(self,  poly: Poly, vals: dict, halfplane: str = None) -> dict[Union[float, sp.Basic]: Union[int, sp.Basic]]:
        log.debug("Starting pole computation for det(G_inv).")

        # 1) Unpacking poly dataclass
        poly_dc = poly # dataclass that needs to be unpacked
        label = poly.label # label for comprehensive logging
        k_var = poly.var  # variable to solve for
        P = poly.poly # sp.Poly polynomial
        deg = poly.degree
        params = poly.free_params
        even = poly.even
        log.info("Computing roots of %s=0.", label)
        log.debug("%s is a %dth order polynomial.", label, deg)
        # Short-circuit for constant:
        if not P.has(k_var):
            # constant in the solve variable -> either identically zero (singular) or no roots
            if sp.cancel(P).equals(0):
                raise ValueError(f"{label} is identically zero; G⁻¹ is singular for all {k_var}.")
            warnings.warn(f"{label} is constant and non-zero in {k_var}; no roots to solve for, returning empty dict.")
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
            log.debug("Even reduction succesful.")
            Q = poly.u_poly
            u = poly.u
            log.debug("Computing roots of the reduced polynomial Q(%s) of order %d.", u, Q.degree())
            assert Q.has(u), "Q should depend on variable u."
            Q_eval = Q.subs(vals)
            leftover_Q = Q_eval.free_symbols - {u}
            assert not leftover_Q, f"Reduced polynomial should have no unknown parameters after eval. Found: {leftover_Q}"
            u_roots = self._poly_roots(Q_eval, u, halfplane=halfplane) # includes halfplane selection
            roots = {}
            for r in u_roots.keys():
                mult = u_roots[r]
                root_val = sp.sqrt(r) # halfplane selection still holds, since sign of Im(sqrt(r)) is the same as sign of Im(r)
                roots[root_val] = roots.get(root_val,0) + mult
                roots[-root_val] = roots.get(-root_val,0) + mult
        ### No even reduction possible; direct approach:
        else:
            log.debug("Even reduction not possible.")
            log.debug("Computing roots of the polynomial P(%s) without even reduction.", k_var)
            roots = self._poly_roots(P_eval, k_var, halfplane=halfplane) # includes halfplane selection
        log.debug("Found %d unique roots.", len(roots))
        log.info("Root computation for %s=0 successful.", label)
        return roots # Roots of denominators/determinants are poles of the k-space Green's function
    # endregion

    # region Real-space
    # -- Symbolic 1D real-space Fourier transform --
    def fourier_entry(self,
                      i: int, j: int,
                      z: float | sp.Basic, z_prime: float | sp.Basic,
                      vals: dict,
                      solve_for: int,
                      z_diff_sign: int = None,
                      lambdified: bool = True):
        log.debug("Computing matrix entry G(z.z')_%d%d.", i, j)
        # 1) Halfplane choice:
        halfplane = self._halfplane_choice(z, z_prime, z_diff_sign=z_diff_sign)
        if halfplane == "coincidence":
            raise ValueError("z and z' must not coincide.")
        if halfplane is None:
            raise ValueError("Choose numbers for z, z' or declare z_diff_sign for the halfplane choice.")
        
        # 2) Poly prep
        solve_for = self._clean_solve_for(solve_for, self.d)
        A = self.adjugate_greens_inverse()
        det_poly_dc = self.determinant_poly(solve_for) # Poly dataclass
        det_poly = det_poly_dc.poly
        det_compiled = self._compile_polynomials(poly=det_poly_dc) # PolyCompiled dataclass
        args_det = det_compiled.args_from_dict(vals)
        k_var = det_poly_dc.var
        ## Cancelling common divisors in A_ij to avoid invalid poles:
        num_poly, denom_poly, A_ij = self._cancel_common_divisors(A[i][j], k_var)
        num_poly_dc, denom_poly_dc = self.numerator_denominator_poly(A_ij,i,j,solve_for) # two Poly dataclasses
        
        # 3) Collecting poles and cancelling with numerator:
        det_poles = self.poly_poles(det_poly_dc, vals, halfplane)
        det_poles_clean = self._cancel_poles_by_numerator(num_poly_dc,det_poles,label="det_poles", vals=vals)
        denom_poles = self.poly_poles(denom_poly_dc, vals, halfplane)
        denom_poles_clean = self._cancel_poles_by_numerator(num_poly_dc,denom_poles,label=f"denom_poles_{i}{j}", vals=vals)
        all_clean_poles = det_poles_clean
        for p in denom_poles_clean.keys():
            m = denom_poles_clean[p]
            all_clean_poles[p] = all_clean_poles.get(p, default=0) + m
        log.debug("There are %d true poles in entry (A/det)_%d%d to contribute to the residue sum.", len(all_clean_poles), i, j)
        
        # 3) Residue sum
        # Short-circuit for no poles present:
        if not all_clean_poles:
            log.debug("No poles found for entry G(z.z')_%d%d. Fourier transform vanishes. Returning 0.")
            return 0
        z_diff = z - z_prime
        phase = sp.exp(sp.I * k_var * z_diff)
        residue_sum = 0
        for n, (k0, m) in enumerate(all_clean_poles):  # pole k0 with their multiplicity m
            # Residue formula for pole of order m:
            # Res = 1/(m-1)! * d^{m-1}/dk^{m-1} [ (k-k0)^m * (numerator(k) / denominator(k) * determinant(k))* phi(k)  ] at k=k0   
            fraction = num_poly.subs(vals) / (denom_poly.subs(vals) * det_poly.subs(vals))
            log.debug("Matrix entry %d%d successfully constructed from polynomial num, denom, det.", i, j)
            first_order = sp.cancel(fraction * (k_var - k0)**m) * phase
            leftover_syms = first_order.free_symbols - set([k_var, z, z_prime])
            assert not leftover_syms, f"Expected no free parameters left after substitution. Got {leftover_syms}."
            if m == 1:
                res = sp.cancel(first_order.subs({k_var: k0}))
            else:
                deriv = sp.cancel(sp.diff(first_order.as_expr(), (k_var, m - 1)))
                res = sp.cancel(first_order.subs({k_var: k0}) / sp.factorial(m - 1))
            log.debug("Residue for pole #%d of order %d computed.", n, m)
            residue_sum += res
            contributed_any = True
            log.debug("res sum snapshot after %d poles: %s", n+1, residue_sum)
        
        # 4) Residue Theorem for the Fourier integral
        fourier_entry = sp.I * residue_sum  # factor of i from residue theorem

        # Optional: Lamdification
        if lambdified:
            fourier_entry = sp.lambdify((z,z_prime), fourier_entry, 'mpmath') # lambdified function of (z,z') for speed and precision
            log.debug("Entry G(z,z')_%d,%d lambdified.", i, j)
        return fourier_entry

    def fourier_transform(self, 
                          z: float | sp.Basic, z_prime: float | sp.Basic,
                          vals: dict,
                          solve_for: int = None,
                          z_diff_sign: int = None,
                          lambdified: bool = True):
        log.info("Fourier transformation of G(k) to G(z.z') started.")
        rows, cols = self.I.shape
        G_zzp = sp.MutableDenseMatrix.zeros(rows, cols)
        for i in range (rows):
            for j in range (cols):
                entry = self.fourier_entry(i,j,z,z_prime,vals,solve_for,z_diff_sign,lambdified)
                G_zzp[i][j] = entry
        log.info("Matrix G(z,z') was successfully computed.")
        G_zzp = G_zzp.as_immutable() # Matrix cannot be changed after this point
        return G_zzp
    
    def rspace_greens_function_last_dim(self, 
                               z: float | sp.Basic, z_prime: float | sp.Basic,
                               vals: dict,
                               z_diff_sign: int = None):
        G_r = self.fourier_transform(z, z_prime, vals, z_diff_sign) 
        return G_r
    
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

    @staticmethod
    def _halfplane_selection(roots: dict, var: sp.Symbol, poly_label: str, halfplane: str = None):
        if halfplane in {"upper", "lower"}:
            im_sign = 1 if halfplane == "upper" else -1
        elif halfplane is not None:
            raise ValueError(f"Unknown halfplane input: {halfplane}. Choose one of ('upper','lower').")
        else:
            im_sign = None
            log.debug("No halfplane specified. Returning all roots.")
            return roots
        root_selection = {}
        for r in roots.keys():
            mult = roots[r]
            if im_sign is not None:
                # Halfplane selection
                try:
                    im = func_timeout(TIMEOUT_GATE, sp.im, args=(r))
                    sgn = func_timeout(TIMEOUT_GATE,sp.sign, args=(im)) 
                    # if this freezes, try an altered version of self._im_sign_of_root:
                    #sgn = self._im_sign_of_root(r, i, n, predicates=predicates, choices=choices) # records ambiguities for symbolic roots
                except FunctionTimedOut:
                    sgn = None
                if sgn not in {-1, 0, 1}:
                    warnings.warn(f"Sign of Im(r) for root r = {r} of {poly_label} could not be determined. Root skipped. Investigate!")
                    # There should not be ambiguities if the roots are not symbolic
                    continue 
                elif sgn.equals(0):
                    raise ValueError(
                        f"Pole at {var}={r} (m={mult}) lies on the real axis; integral is ill-defined. Provide finite broadening η.")
                elif sgn in {-1, 1} and sgn != im_sign:
                    log.debug("Pole %s=%s (m=%s) of %s lies in wrong half-plane; skipped.", var, r, mult, poly_label)
                    continue
                else:
                    pass # continue with adding this root the dict, since sgn==im_sign 
            root_selection[r] = mult
        log.debug("Halfplane selection for %s successful.", poly_label)
        return root_selection
    
    @staticmethod
    def _halfplane_choice(z: sp.Symbol | float, z_prime: sp.Symbol | float, z_diff_sign: int = None) -> str | None:
        # numeric-only decision; returns +1, -1, 0, or None (unknown)
        if z.is_real is not True or z_prime.is_real is not True:
            raise ValueError("Both z and z′ must be real numbers or real symbols.")
        if z.is_number and z_prime.is_number:
            if z > z_prime:  return "upper"
            if z < z_prime:  return "lower"
            return "coincidence"
        if z_diff_sign is not None: 
            if z_diff_sign > 0: return "upper"
            if z_diff_sign < 0: return "lower"
            raise ValueError(f"Expected z_diff_sign from (1,-1,None). Got {z_diff_sign}.")
        return None
    
    def _cancel_common_divisors(self, ratio: sp.Basic, var: sp.Symbol) -> tuple[sp.Poly,sp.Poly,sp.Expr]:
        num, den = sp.fraction(ratio)
        Pnum = self._poly_in(var, num.as_expr())
        Pden = self._poly_in(var, den.as_expr())
        gcd = sp.gcd(Pnum, Pden) 
        if gcd.has(var):
            # sp.Poly objects returned; Reduced polynomials
            Pnum = Pnum.quo(gcd) 
            Pden = Pden.quo(gcd)
        reduced_ratio = Pnum.as_expr() / Pden.as_expr()
        reduced_ratio = sp.cancel(reduced_ratio)
        return Pnum, Pden, reduced_ratio.as_expr()
    
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
            #even = poly_dc.even
            params = poly_dc.free_params
            #if even:
            #    u = poly_dc.u # squared variable
            #    u_poly = poly_dc.u_poly # reduced polynomial for u, sp.Poly object
        elif var is None:
            raise ValueError("No variable provided. If poly is not a Poly dataclass, 'var' needs to be provided.")
        elif isinstance(poly, Union[sp.Expr, sp.Basic]):
            poly = poly.as_expr()
            poly = self._poly_in(var, poly) # sp.Poly object now
        if isinstance(poly,sp.Poly):
            k = var
            deg, even, params, u, u_poly = self._gather_poly_metadata(poly, k)
            P = poly # consistent naming between cases

        ## Try even reduction
        #if even:
        #    assert u is not None, "for an even polynomial, a squared variable should have been generated."
        #    Q = u_poly
        #    assert Q.has(u), "Q should depend on variable u."
        #    dQ_du = Q.diff(u) # still sp.Poly 
        #    dP_dk = sp.cancel(2 * k * dQ_du.subs({u: k**2}))
        #else:
        #    dQ_du = None
        #    dP_dk = P.diff(k)

        # Lambdify P and P'
        P = sp.lambdify((k, *params), P, modules=modules)
        #dP_dk = sp.lambdify((k, *params), dP_dk, modules=modules)
        #if Q is not None:
        #    Q = sp.lambdify((u, *params), Q, modules=modules)
        #    dQ_du = sp.lambdify((u, *params), dQ_du, modules=modules)

        return PolyCompiled(
            var=k,
            params=params,
            backend=backend,
            prec=prec,
            #even=even,
            P=P,
            #Pprime=dP_dk,
            #u=u,
            #Q=Q,
            #Qprime=dQ_du,
        )

    def _cancel_poles_by_numerator(self, numerator: Poly | sp.Poly, roots: dict, label: str, vals: dict, 
                                   var: sp.Symbol = None, 
                                   *,
                                   prec: int = 80,            # working precision for numeric tests
                                   atol: float = 1e-28,       # absolute tolerance
                                   rtol: float = 1e-12        # relative tolerance (scale-aware)
                                   ):
        """
        For each candidate pole r with denominator multiplicity m_den, compute the
        numerator multiplicity m_num at r by derivative counting and return the
        updated multiplicity max(m_den - m_num, 0). Remove entries that cancel.

        Returns:
        - dict in  -> dict out   {root: updated_mult}
        """
        # 0) Sanitize input
        if not roots:
            return{}
        if isinstance(numerator, Poly):
            poly = numerator.poly # sp.Poly object
            var = numerator.var
        elif isinstance(numerator, sp.Poly):
            if var is None:
                raise ValueError("Variable input needed, if poly is not a Poly dataclass object.")
            poly = numerator
        else:
            raise TypeError(f"numerator needs to be a Poly dataclass object or sp.Poly. Got {type(numerator)}.")

        # 1) Maximum multiplicity for maximum derivative order needed:
        max_needed = max((int(m) for m in roots.values()), default=0)
        if max_needed < 1:
            warnings.warn(f"Roots dict '{label}' is empty.")
            return {} 

        # 2) Build and compile derivatives 0..max_needed and lambdify once (mpmath backend):
        num_compiled = self._compile_polynomials(numerator, var=var) # 0th order compilation
        args_num = num_compiled.args_from_dict(vals) # arguments needed for numerical evaluation
        derivs = [poly.diff(var, n) for n in range(0, max_needed+1)] # collects all derivatives needed as sp.Poly object, incl. 0th order
        derivs_compiled = [self._compile_polynomials(deriv, var=var) for deriv in derivs]  # f^(0), f^(1), ..., f^(max_needed)

        # helper: robust zero test for order j using next derivative as scale
        def vanishes_at(diff_order: int, root: sp.Symbol) -> bool:
            funcs = derivs_compiled
            order = diff_order
            f = funcs[order] # lambdified function requiring args input and var
            with mp.workdps(prec):
                val = f(root, *args_num) 
                # Use next derivative magnitude as local scale (if available)
                if order + 1 < len(funcs):
                    val_scale = funcs[order + 1](root, *args_num)
                    scale = abs(val_scale) + 1
                else:
                    scale = 1
                thr = max(atol, rtol * scale)
                return abs(val) <= thr

        # --- process each root ---
        updated_roots = {}
        with mp.workdps(prec):
            for r, m_den in roots.items():
                m_den = int(m_den)
                m_num = 0
                # Count how many derivatives vanish at r (up to m_den is enough)
                for j in range(m_den):
                    if vanishes_at(j, r):
                        m_num += 1
                        if m_num >= m_den:  # can't reduce below zero anyway
                            break
                    else:
                        break

                m_eff = max(m_den - m_num, 0)
                if m_eff > 0:
                    updated_roots[r] = m_eff
                # else: fully cancelled → omit
        return updated_roots
        
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
    
    def _poly_roots(self, poly: sp.Poly, var: sp.Symbol, poly_label: str, halfplane: str = None) -> dict:
        # Garantee univariate polynomial:
        poly = sp.Poly(poly, var, domain="EX")
        if poly.degree() <= 0:
            warnings.warn(f"{poly_label} constant in {var}. Should have triggered earlier. Returning empty dict.")
            return {}
        # First attempt: exact root solving directly with poly.roots()
        try:
            log.debug("Attempting exact root solving with poly.roots() for %s.", poly_label)
            roots = func_timeout(TIMEOUT_GATE, poly.roots)
            log.debug("Direct root solving successful for %s.", poly_label)
            root_selection = self._halfplane_selection(roots, var, poly_label=poly_label, halfplane=halfplane)
            return root_selection
        except FunctionTimedOut:
            warnings.warn(f"Exact poly.roots() solving exceeded {TIMEOUT_GATE} seconds for {poly_label}")
        # Second attempt: exact root solving indirectly via factorization
        try: 
            log.debug("Attempting root solving with factorization for %s.", poly_label)
            # Square-free decomposition:
            _, factors = func_timeout(TIMEOUT_GATE,poly.sqf_list) # factors is a list of (factor, multplicity) tuples
            log.debug("Square-free decomposition of poly successful for %s.", poly_label)
            roots = {}
            for f, mult in factors:
                f_roots = func_timeout(TIMEOUT_GATE,f.roots)
                for r in f_roots.keys():
                    m = f_roots[r]*mult # update multiplicity with decomposition factor
                    roots[r] = roots.get(r, 0) + m
            log.debug("Root solving via factorization successful for %s.", poly_label) 
            root_selection = self._halfplane_selection(roots, var, poly_label=poly_label, halfplane=halfplane)
            return root_selection
        except FunctionTimedOut:
            warnings.warn(f"Root solving with factorization exceeded {TIMEOUT_GATE} seconds for {poly_label}")
        # Third attempt: numerical root solving with poly.nroots()
        try:
            log.debug("Attempting numerical root solving with poly.nroots() for %s.", poly_label)
            # Numerical root approximation before clustering:
            poly_monic = poly.monic() # divides the polynomial by its leading coefficient
            roots_list = func_timeout(TIMEOUT_GATE,poly_monic.nroots) # returns list of roots as float objects
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
            log.debug("Clustering of root approximations successfull for %s. Roots dict returned.", poly_label)
            root_selection = self._halfplane_selection(roots, var, poly_label=poly_label, halfplane=halfplane)
            return root_selection
        except FunctionTimedOut:
            raise RuntimeError(f"All attempts at root solving for {poly_label} exceeded {TIMEOUT_GATE} seconds. Investigate.")
       
    def _im_sign_of_root(self, k0, i, n, predicates=None, choices=None):
        """
        DEPRECATED METHOD 
        KEPT ONLY FOR A SAFETY NET
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
    # endregion
