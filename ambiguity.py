from dataclasses import dataclass, field
from typing import Any, List, Optional
import sympy as sp

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