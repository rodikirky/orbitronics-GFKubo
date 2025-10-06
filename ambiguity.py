'''Ambiguity ledger (what it is, how to use it)
--------------------------------------------
The greens function calculator records places where a symbolic decision could not be made
unambiguously (e.g., undecidable sign(Im k₀), non-polynomial λᵢ(k), ill-conditioned
eigenbasis). Each public compute method calls `_reset_ambiguities()` on entry and
adds entries via `_add_amb(...)`.

• Inspect: after a compute call, use `get_ambiguities()` to obtain a list of
  `Ambiguity` items (see ambiguity.py). Call `format_ambiguities()` for a
  human-readable summary.

• Typical use: if ambiguities are present, present the options to the user or
  re-run with additional assumptions (e.g., SymPy predicates like
  `sp.Q.positive(omega - V_F)`) or with an explicit `z_diff_sign`. The ledger
  is the programmatic source of truth for what needs disambiguation; logging
  is only informational.

API:
  - self._reset_ambiguities()    # clear ledger (done automatically)
  - self._add_amb(**fields)      # internal use at ambiguity hotspots
  - self.get_ambiguities() -> list[Ambiguity]
  - self.format_ambiguities() -> str
  '''
from dataclasses import dataclass, field
import sympy as sp
from typing import Any, List, Optional, Dict, Tuple, Union
ChoiceKey = Tuple[str, str]
CaseAssumptions = Union[
    List[sp.Basic],
    Dict[str, Any],   # expects keys: "predicates": List[sp.Basic], "choices": Dict[Tuple[str,str], Any]
]

@dataclass
class Ambiguity:
    where: str
    what: str
    predicate: Optional[sp.Basic]
    options: List[Any]
    consequence: str
    data: dict = field(default_factory=dict)
    severity: str = "warn"  # "info" | "warn" | "error"

class AmbiguityLedger:
    def __init__(self): self._items: List[Ambiguity] = []
    def reset(self): self._items.clear()
    def add(self, **kw): self._items.append(Ambiguity(**kw))
    def items(self) -> List[Ambiguity]: return list(self._items)
    def format(self) -> str:
        lines=[]
        for a in self._items:
            opts = ", ".join(map(str, a.options))
            lines.append(
                f"[{a.severity}] {a.where}: {a.what}\n"
                f"  ⇒ {a.consequence}\n"
                f"  Options: {opts}\n"
                f"  Data: {a.data}"
            )
        return "\n".join(lines)
    
class AggregatedAmbiguityError(RuntimeError):
    """Raised when ambiguities were recorded but not resolved before returning a result."""
    def __init__(self, message: str, items=None):
        super().__init__(message)
        self.items = items or []

def build_case_assumptions_from_ledger(
    items: List[Ambiguity],
    *,
    default_halfplane: int | None = None,   # +1 (upper) / -1 (lower); if None, don't auto-choose
    accept_condition_sets: bool = True,     # True = accept ConditionSet when solver can’t close form
    prefer_constant_poly_edgecase: bool | None = None  # None = leave unresolved
) -> dict:
    """
    Convert recorded Ambiguity items into a `case_assumptions` dict that
    `GreensFunctionCalculator` understands: {"predicates": [...], "choices": {...}}.

    Policy:
    - If an item carries a concrete `predicate`, include it verbatim.
    - For known choice sites, fill `choices[(where, what)]` with reasonable defaults
      unless the user wants to decide later (by leaving them unset).

    Known choice keys in your solver:
      ("residue", f"lambda_{i}.sign_im_root")  -> expects +1 or -1
      ("roots",   f"lambda_{i}.condition_set") -> expects "ConditionSet" or "FiniteSet"
      ("roots",   f"lambda_{i}.poly_constant_yet_not") -> expects "constant" or "not-constant"
    """
    predicates: List[sp.Basic] = []
    choices: Dict[ChoiceKey, Any] = {}

    for a in items:
        # 1) Take explicit predicates when provided
        if a.predicate is not None:
            predicates.append(a.predicate)

        # 2) Fill known choice slots
        key: ChoiceKey = (a.where, a.what)

        # Residue half-plane ambiguity
        if a.where == "residue" and a.what.endswith(".sign_im_root"):
            if default_halfplane in (+1, -1):
                choices[key] = int(default_halfplane)
            # else leave unset to force the user to decide

        # Root solving returned a ConditionSet; choose to accept or insist on tightening
        elif a.where == "roots" and a.what.endswith(".condition_set"):
            choices[key] = "ConditionSet" if accept_condition_sets else "FiniteSet"

        # Polynomial deg==0 edge case
        elif a.where == "roots" and a.what.endswith(".poly_constant_yet_not"):
            if prefer_constant_poly_edgecase is True:
                choices[key] = "constant"
            elif prefer_constant_poly_edgecase is False:
                choices[key] = "not-constant"
            # else leave unset

        # Future: fall through keeps unknown keys visible but undecided

    # Deduplicate predicates while preserving order
    seen = set()
    uniq_predicates = []
    for p in predicates:
        if sp.srepr(p) not in seen:
            uniq_predicates.append(p)
            seen.add(sp.srepr(p))

    return {"predicates": uniq_predicates, "choices": choices}

