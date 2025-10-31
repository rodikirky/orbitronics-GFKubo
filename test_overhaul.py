from greens import GreensFunctionCalculator
from system import OrbitronicHamiltonianSystem
import numpy as np
import sympy as sp
from utils import sanitize_vector, to_jsonable
#from ambiguity import AggregatedAmbiguityError
from pathlib import Path
import json
import pickle
import logging
logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.captureWarnings(True) # optional: route warnings -> logging
log = logging.getLogger(__name__)

#######################################
# Orbitronics system in symbolic mode
#######################################
symbolic_mode = True
mass = sp.symbols("m", real=True, positive=True)
orbital_texture_coupling = sp.symbols("gamma", real=True)
#orbital_texture_coupling = 0
exchange_interaction_coupling = sp.symbols("J", real=True)
#exchange_interaction_coupling = 0 # with J=0 determinant is an even polynomial in k
mag1, mag2, mag3 = sp.symbols("M_1 M_2 M_3", real=True)
magnetisation = sanitize_vector([mag1, mag2, mag3], symbolic=symbolic_mode)

system3D = OrbitronicHamiltonianSystem(mass=mass,
                                       orbital_texture_coupling=orbital_texture_coupling,
                                       exchange_interaction_coupling=exchange_interaction_coupling,
                                       magnetisation=magnetisation,
                                       symbolic=symbolic_mode)
def hamiltonian(momentum):
    H_k = system3D.get_hamiltonian(momentum)
    return H_k
