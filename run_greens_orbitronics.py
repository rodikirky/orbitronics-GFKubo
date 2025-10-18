from __future__ import annotations
import argparse
from pathlib import Path
import sympy as sp
from runner_utils import setup_logging, validated_k

# Project imports
from greens import GreensFunctionCalculator
from system import OrbitronicHamiltonianSystem
from utils import sanitize_vector, save_result

# numeric defaults
DEFAULT_M = [0.0, 0.0, 1.0] # Magnetisation vector
DEFAULT_OMEGA = 1.5
DEFAULT_ETA = 1e-3
DEFAULT_MASS = 2.0
DEFAULT_GAMMA = 0.4 
DEFAULT_J = 0.0

# symbolic defaults
SYM_M = list(sp.symbols("M_1 M_2 M_3", real=True)) # Magnetisation vector
SYM_OMEGA = sp.symbols("omega", real=True)
SYM_ETA = sp.symbols("eta", real=True, positive=True)
SYM_MASS = sp.symbols("m", real=True, positive=True)
SYM_GAMMA = sp.symbols("gamma", real=True)
SYM_J = sp.symbols("J", real=True)


def build_calc_orbitronics(symbolic: bool, omega=None, eta=None, m=None, gamma=None, J=None, M=None):
    if symbolic:
        omega = SYM_OMEGA if omega is None else omega
        eta = SYM_ETA if eta is None else eta
        m = SYM_MASS if m is None else m
        gamma = SYM_GAMMA if gamma is None else gamma
        J = SYM_J if J is None else J
        M = SYM_M if M is None else M
    else:
        omega = DEFAULT_OMEGA if omega is None else omega
        eta = DEFAULT_ETA if eta is None else eta
        m = DEFAULT_MASS if m is None else m
        gamma = DEFAULT_GAMMA if gamma is None else gamma
        J = DEFAULT_J if J is None else J
        M = DEFAULT_M if M is None else M
    magnetisation = sanitize_vector(M, symbolic=symbolic)
    sys = OrbitronicHamiltonianSystem(m, gamma, J, magnetisation, symbolic=symbolic)
    def H(k): return sys.get_hamiltonian(k)
    I = sys.identity
    return GreensFunctionCalculator(H, I, symbolic, omega, eta, retarded=True, dimension=3)

def cmd_ginv(args):
    calc = build_calc_orbitronics(args.symbolic, args.omega, args.eta, args.m, args.gamma, args.J, args.M)
    k = None if args.symbolic else validated_k(args.k, 3)
    if not args.symbolic and k is None:
        raise ValueError("Numeric mode requires --k (one for d=1; else d values).")#
    if args.symbolic and args.k is not None:
        raise ValueError("Symbolic mode does not accept --k; uses default k symbols.")
    Ginv = calc.get_greens_inverse(k)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    save_result(Ginv, Path(args.out)/f"{args.name}_Ginv", symbolic=args.symbolic)

def cmd_gk(args):
    calc = build_calc_orbitronics(args.symbolic, args.omega, args.eta, args.m, args.gamma, args.J, args.M)
    k = None if args.symbolic else validated_k(args.k, 3)
    if not args.symbolic and k is None:
        raise ValueError("Numeric mode requires --k (one for d=1; else d values).")#
    if args.symbolic and args.k is not None:
        raise ValueError("Symbolic mode does not accept --k; uses default k symbols.")
    Gk = calc.compute_kspace_greens_function(k)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    save_result(Gk, Path(args.out)/f"{args.name}_G_k", symbolic=args.symbolic)

def cmd_gz_sym(args):
    calc = build_calc_orbitronics(True, args.omega, args.eta, args.m, args.gamma, args.J, args.M)
    Gr = calc.compute_rspace_greens_symbolic_1d(args.z, args.zprime, full_matrix=args.full_matrix)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    save_result(Gr, Path(args.out)/f"{args.name}_G_r", symbolic=True)

def main():
    p = argparse.ArgumentParser(
        prog="run_greens_orbitronics",
        description="Orbitronics 3D Green’s-function runner (presets).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--log", default="INFO",
                   choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"])
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        sp.add_argument("--symbolic", action="store_true")
        sp.add_argument("--omega", type=float, default=DEFAULT_OMEGA)
        sp.add_argument("--eta", type=float, default=DEFAULT_ETA)
        sp.add_argument("--m", type=float, default=DEFAULT_MASS)
        sp.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
        sp.add_argument("--J", type=float, default=DEFAULT_J)
        sp.add_argument("--M", type=float, nargs=3, default=DEFAULT_M)
        sp.add_argument("--out", default="results/greens_runs")
        sp.add_argument("--name", default="run")

    sp1 = sub.add_parser("ginv", help="Compute G^{-1}(k,ω) [orbitronics]")
    add_common(sp1); sp1.add_argument("--k", type=float, nargs=3, help="k_x k_y k_z (numeric mode).")
    sp1.set_defaults(func=cmd_ginv)

    sp2 = sub.add_parser("gk", help="Compute G(k,ω) [orbitronics]")
    add_common(sp2); sp2.add_argument("--k", type=float, nargs=3, help="k_x k_y k_z (numeric mode).")
    sp2.set_defaults(func=cmd_gk)

    sp3 = sub.add_parser(
        "gz-sym",
        help="Symbolic 1D G(z,z';ω) for the orbitronics model: integrates along the last k-component."
    )
    add_common(sp3)
    sp3.add_argument("--z", type=float, required=True, help="Observation point z.")
    sp3.add_argument("--zprime", type=float, required=True, help="Source point z′.")
    sp3.add_argument("--full-matrix", action="store_true",
                    help="Return full matrix (default: diagonal only).")
    sp3.set_defaults(func=cmd_gz_sym)

    args = p.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.log, out_dir, args.name)
    args.func(args)

if __name__ == "__main__":
    main()