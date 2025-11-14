# orbitronics-GFKubo
orbitronics computations using the Kubo formula with Green's functions for a layered system consisting of a non-ferromagnetic material on one side, an infinitesimally small interface and a ferromagnetic material on the other side.
<img width="1279" height="315" alt="grafik" src="https://github.com/user-attachments/assets/728fb2bb-5a02-44bf-97af-cfc6eb2800a1" />

# Repository structure (THIS IS THE AIM; still to be sorted)
orbitronics-gfkubo/
├─ src/orbitronics_gfkubo/
│  ├─ __init__.py          # package init, version, main imports
│  ├─ greens.py            # GreensFunctionCalculator and related helpers
│  ├─ system.py            # Hamiltonian system classes and builders
│  ├─ utils.py             # generic utilities (logging, type helpers, etc.)
│  └─ config.py            # optional: default parameters, paths, constants
│
├─ scripts/
│  ├─ run_example_1d_chain.py
│  └─ run_bulk_greens_3d.py
│
├─ tests/
│  ├─ test_greens.py
│  └─ test_system.py
│
├─ examples/
│  ├─ minimal_1d_greens.ipynb
│  └─ interface_greens_2d.ipynb
│
├─ docs/
│  ├─ derivations.md
│  └─ design_notes.md
│
├─ pyproject.toml
├─ README.md
└─ LICENSE
