import sympy as sp
import mpmath as mp
from collections.abc import Mapping

def cancel_poles_by_numerator(
    roots: dict, 
    label: str,
    numerator: sp.Poly,
    var: sp.Symbol,  
    *,
    prec: int = 80,            # working precision for numeric tests
    atol: float = 1e-28,       # absolute tolerance
    rtol: float = 1e-12        # relative tolerance (scale-aware)
):
    """
    For each candidate pole r with denominator multiplicity m_den, compute the
    numerator multiplicity m_num at r by derivative counting and return the
    updated multiplicity max(m_den - m_num, 0). Remove entries that cancel.

    Returns the SAME container type as provided:
      - dict in  -> dict out   {root: updated_mult}
      - list/iterable in -> list of (root, updated_mult)
    """
    assert isinstance(numerator, sp.Poly), f"Expected sp.Poly type for numerator, got {type(numerator)}."

    # 1) Maximum multiplicity for maximum derivative needed:
    max_needed = max((int(m) for m in roots.values()), default=0)
    if max_needed < 1:
        warnings.warn(f"Roots dict '{label}' is empty.")
        return {} 

    # 2) Build derivatives 0..max_needed and lambdify once (mpmath backend):
    expr = numerator.as_expr()
    deriv_exprs = [expr] # collects all derivatives needed
    for _ in range(max_needed-1):
        deriv = sp.diff(deriv_exprs[-1], var)
        deriv_exprs.append(deriv.as_expr())
    funs = [sp.lambdify(k, e, "mpmath") for e in deriv_exprs]  # f^(0), f^(1), ..., f^(max_needed)

    # helper: robust zero test for order j using next derivative as scale
    def vanishes_at(order_j: int, r) -> bool:
        f = funs[order_j]
        with mp.workdps(prec):
            v = f(r)
            # Use next derivative magnitude as local scale (if available)
            if order_j + 1 < len(funs):
                vp = funs[order_j + 1](r)
                scale = abs(vp) + 1
            else:
                scale = 1
            thr = max(atol, rtol * scale)
            return abs(v) <= thr

    # --- process each root ---
    updated = {}
    with mp.workdps(prec):
        for r, m_den in roots_dict.items():
            m_den = int(m_den)
            m_num = 0
            # Count how many derivatives vanish at r (up to m_den is enough)
            for j in range(min(m_den, max_needed) + 1):  # +1 lets us detect exact m_den cases
                if vanishes_at(j, r):
                    m_num += 1
                    if m_num >= m_den:  # can't reduce below zero anyway
                        break
                else:
                    break

            m_eff = max(m_den - m_num, 0)
            if m_eff > 0:
                updated[r] = m_eff
            # else: fully cancelled → omit

    # --- return in original container type ---
    if in_is_dict:
        return updated
    else:
        return [(r, updated[r]) for r in roots_dict if r in updated]
