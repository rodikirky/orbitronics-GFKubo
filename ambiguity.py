'''
Ambiguity ledger (what it is, how to use it)
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

def build_case_from_ledger(
    items: List[Ambiguity]
) -> dict:
    """
    Convert recorded Ambiguity items into a `case_assumptions` dict that
    `GreensFunctionCalculator` understands: {"predicates": [...], "choices": {...}}.
    """
    predicates: List[sp.Basic] = []
    choices: Dict[ChoiceKey, Any] = {}



