from pathlib import Path
import json
import pickle
import sympy as sp

# --- Safe SymPy eval for srepr strings ---
_SYMPY_NS = {name: getattr(sp, name) for name in dir(sp) if not name.startswith("_")}

def _from_srepr_maybe(s: str):
    """
    Try to interpret string 's' as a SymPy srepr() string.
    If successful, return a SymPy object; otherwise return the original string.
    """
    try:
        obj = eval(s, {"__builtins__": {}}, _SYMPY_NS)  # locked namespace
        return obj if isinstance(obj, sp.Basic) else s
    except Exception:
        return s

def _from_jsonable(obj):
    """
    Inverse of your 'to_jsonable' (best effort):
    - strings that look like SymPy srepr get turned back into SymPy objects
    - lists/tuples/sets were serialized as lists; we keep them as lists
    - dict keys that were SymPy become SymPy again (via srepr strings)
    """
    if isinstance(obj, dict):
        # Rebuild keys & values
        new = {}
        for k, v in obj.items():
            k2 = _from_srepr_maybe(k) if isinstance(k, str) else k
            new[k2] = _from_jsonable(v)
        return new
    if isinstance(obj, list):
        return [_from_jsonable(x) for x in obj]
    if isinstance(obj, str):
        return _from_srepr_maybe(obj)
    # ints/floats/bools/None pass through
    return obj

# --- Public loader API ---
def load_roots(json_path: Path = Path("results/toy_multi_roots.json")):
    """
    Load the JSON file produced by json.dump(to_jsonable(roots), ...),
    reconstructing SymPy keys/values where possible.
    """
    with json_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return _from_jsonable(raw)

def load_free_syms(pkl_path: Path = Path("results/toy_multi_free_symbols.pkl")):
    """
    Load the pickled SymPy object 'free_symbols'.
    """
    with pkl_path.open("rb") as f:
        return pickle.load(f)

# --- Example usage ---
if __name__ == "__main__":
    roots = load_roots()
    free_symbols = load_free_syms()

    print("Loaded objects:")
    print(" - type(roots):", type(roots))
    print(" - len(roots):", len(roots) if hasattr(roots, "__len__") else "n/a")
    print(" - type(det):  ", type(free_symbols))

    # If det is a SymPy expression, you can inspect or evaluate it:
    if isinstance(free_symbols, list):
        print("free symbols:", free_symbols)
    
    eta = free_symbols[0]
    k_y = free_symbols[1]
    J = free_symbols[2]
    k_x = free_symbols[3]
    k_z = free_symbols[4]
    M_3 = free_symbols[5]
    omega = free_symbols[6]
    gamma = free_symbols[7]

    # If roots is a dict keyed by SymPy expressions, show a small sample:
    if isinstance(roots, dict) and roots:
        sample_key = next(iter(roots.keys()))
        print("Sample roots key type:", type(sample_key))
        print("Sample value type:", type(roots[sample_key]))
        print("Free symbols in sample key:", sample_key.free_symbols)

    #if isinstance(sample_key, sp.Basic):
    #    sample_key_zero = sample_key.subs({k_x: 0, k_y: 0, k_z: 0})
    #    print("sample_key for kx=ky=kz=0:", sp.cancel(sample_key_zero))
        
    
