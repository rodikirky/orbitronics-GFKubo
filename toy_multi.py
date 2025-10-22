from greens import GreensFunctionCalculator
from system import OrbitronicHamiltonianSystem
import numpy as np
import sympy as sp
from utils import sanitize_vector, print_symbolic_matrix, invert_matrix, sanitize_matrix
#from ambiguity import AggregatedAmbiguityError
import logging
logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.captureWarnings(True) # optional: route warnings -> logging
log = logging.getLogger(__name__)

################################################################
# Multi-channel toy model in symbolic mode
################################################################
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

I = system3D.identity
omega = sp.symbols("omega", real=True)
eta = sp.symbols("eta", real=True, positive=True)

vals = {
    #omega: 0.8, 
    #orbital_texture_coupling: 0.3, 
    #exchange_interaction_coupling: 1.0,
    mag1: 0.0, mag2: 0.0, #mag3: 3.0,
    mass: 1.0, 
    #eta: 1e-6,      # small positive η for retarded GF
    #k_x: 0.1, k_y: -0.2
}

greenscalculator = GreensFunctionCalculator(
    hamiltonian=hamiltonian,
    identity=I,
    symbolic=symbolic_mode,
    energy_level=omega,
    broadening=eta,
    retarded=True,
    dimension=3)

G_inv = greenscalculator.get_greens_inverse()
G_inv = G_inv.subs(vals)
#print("G^{-1}(k,ω):", G_inv)
#G_inv_num = G_inv.subs(vals)
#print("G^{-1}(k,ω) after substitution:", G_inv_num)
#print_symbolic_matrix(G_inv, name="G^{-1}(k,ω)")
#print("Type of G_inv before sanitization:", type(G_inv))
#G_k = G_inv.inv(method='LU') if symbolic_mode else np.linalg.inv(G_inv) # the method choice was the game changer
#print_symbolic_matrix(G_k, name="G(k,ω)")
#G_k = invert_matrix(G_inv, symbolic=symbolic_mode)
#G_k = greenscalculator.compute_kspace_greens_function() 
#print_symbolic_matrix(G_k, name="G(k,ω)")
roots = greenscalculator.compute_roots_greens_inverse(vals=vals) 
print("Number of roots of det(G^{-1}(k,ω))=0:", len(roots))
#roots_unique = list(roots.keys())
#root_1 = roots_unique[0]
#k_var = greenscalculator.k_symbols[-1]
det = sp.cancel(greenscalculator._determinant(G_inv))
print("Free symbols in det(G_inv):", det.free_symbols)

