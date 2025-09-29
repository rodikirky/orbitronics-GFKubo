# Ambiguity ledger

Purpose: record undecidable symbolic choices (e.g., sign(Im k₀), non-polynomial λᵢ(k), ill-conditioned eigenbasis).

API:
- `GreensFunctionCalculator.get_ambiguities()` → list[Ambiguity]
- `AmbiguityLedger.format()` → human summary

Typical resolution:
- Add SymPy assumptions (e.g., `sp.Q.positive(omega - V_F)`), or
- Provide `z_diff_sign`, or
- Switch to numeric fallback.

Returned fields: `where`, `what`, `predicate`, `options`, `consequence`, `data`, `severity`.
