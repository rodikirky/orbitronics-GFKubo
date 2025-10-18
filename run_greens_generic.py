from __future__ import annotations
import argparse, json
from pathlib import Path
from runner_utils import setup_logging, validated_k, load_builder

# Project imports
from greens import GreensFunctionCalculator
from utils import save_result

# numeric defaults
DEFAULT_M = [0.0, 0.0, 1.0] # Magnetisation vector
DEFAULT_OMEGA = 1.5
DEFAULT_ETA = 1e-3
DEFAULT_MASS = 2.0
DEFAULT_GAMMA = 0.4 
DEFAULT_J = 0.0

def cmd_ginv(args):
    build = load_builder(args.builder)
    kw = json.loads(args.builder_kwargs) if args.builder_kwargs else {}
    H, I, d = build(symbolic=args.symbolic, **kw)
    calc = GreensFunctionCalculator(H, I, args.symbolic, args.omega, args.eta,
                                    retarded=not args.advanced, dimension=d)
    k = None if args.symbolic else validated_k(args.k, d)
    if not args.symbolic and k is None:
        raise ValueError("Numeric mode requires --k.")
    Ginv = calc.get_greens_inverse(k)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    save_result(Ginv, Path(args.out)/f"{args.name}_Ginv", symbolic=args.symbolic)

def cmd_gk(args):
    build = load_builder(args.builder)
    kw = json.loads(args.builder_kwargs) if args.builder_kwargs else {}
    H, I, d = build(symbolic=args.symbolic, **kw)
    calc = GreensFunctionCalculator(H, I, args.symbolic, args.omega, args.eta,
                                    retarded=not args.advanced, dimension=d)
    k = None if args.symbolic else validated_k(args.k, d)
    if not args.symbolic and k is None:
        raise ValueError("Numeric mode requires --k.")
    G = calc.compute_kspace_greens_function(k)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    save_result(G, Path(args.out)/f"{args.name}_G", symbolic=args.symbolic)

def cmd_gz_sym(args):
    build = load_builder(args.builder)
    kw = json.loads(args.builder_kwargs) if args.builder_kwargs else {}
    H, I, d = build(symbolic=True, **kw)
    calc = GreensFunctionCalculator(H, I, True, args.omega_sym, args.eta_sym,
                                    retarded=not args.advanced, dimension=d)
    Gr = calc.compute_rspace_greens_symbolic_1d(args.z, args.zprime, full_matrix=args.full_matrix)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    save_result(Gr, Path(args.out)/f"{args.name}_G_r", symbolic=True)

def main():
    p = argparse.ArgumentParser(
        prog="run_greens_generic",
        description="Generic Green’s-function runner (external builder).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--log", default="INFO",
                   choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"])
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        sp.add_argument("--symbolic", action="store_true")
        sp.add_argument("--omega", type=float, default=DEFAULT_OMEGA)
        sp.add_argument("--eta", type=float, default=DEFAULT_ETA)
        sp3.add_argument("--omega-sym", type=float, default=None)
        sp3.add_argument("--eta-sym", type=float, default=None)
        sp.add_argument("--advanced", action="store_true",
                        help="Use advanced (−iη) instead of retarded (+iη).")
        sp.add_argument("--out", default="results/greens_runs")
        sp.add_argument("--name", default="run")
        sp.add_argument("--builder", required=True,
                        help="Import path 'module.submodule:func' → (H, I, dimension).")
        sp.add_argument("--builder-kwargs", default="",
                        help="JSON kwargs for builder, e.g. '{\"m\":2.0}'.")

    sp1 = sub.add_parser("ginv", help="Compute G^{-1}(k,ω)")
    add_common(sp1); sp1.add_argument("--k", type=float, nargs="+")
    sp1.set_defaults(func=cmd_ginv)

    sp2 = sub.add_parser("gk", help="Compute G(k,ω)")
    add_common(sp2); sp2.add_argument("--k", type=float, nargs="+")
    sp2.set_defaults(func=cmd_gk)

    sp3 = sub.add_parser("gz-sym", help="Symbolically compute 1D G(z, z';ω)")
    add_common(sp3)
    sp3.add_argument("--z", type=float, required=True)
    sp3.add_argument("--zprime", type=float, required=True)
    sp3.add_argument("--full-matrix", action="store_true")
    sp3.set_defaults(func=cmd_gz_sym)

    args = p.parse_args()
    out_dir = Path(getattr(args, "out", "results/greens_runs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.log, out_dir, getattr(args, "name", "run"))
    args.func(args)

if __name__ == "__main__":
    main()