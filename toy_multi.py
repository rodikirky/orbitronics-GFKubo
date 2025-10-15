from greens import GreensFunctionCalculator
from system import OrbitronicHamiltonianSystem
import numpy as np
import sympy as sp
from utils import sanitize_vector
#from ambiguity import AggregatedAmbiguityError
import logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.captureWarnings(True) # optional: route warnings -> logging
log = logging.getLogger(__name__)

################################################################
# Multi-channel toy model in symbolic mode
################################################################
symbolic_mode = True
mass = sp.symbols("m", real=True, positive=True)
orbital_texture_coupling = sp.symbols("gamma", real=True)
exchange_interaction_coupling = sp.symbols("J", real=True)
mag1, mag2, mag3 = sp.symbols("M_1 M_2 M_3", real=True)
magnetisation = sanitize_vector([mag1, mag2, mag3], symbolic=symbolic_mode)

system3D = OrbitronicHamiltonianSystem(mass=mass,
                                       orbital_texture_coupling=orbital_texture_coupling,
                                       exchange_interaction_coupling=exchange_interaction_coupling,
                                       magnetisation=magnetisation,
                                       symbolic=symbolic_mode,
                                       verbose=True)

                