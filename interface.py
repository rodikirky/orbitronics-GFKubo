from utils import invert_matrix
import sympy as sp
from typing import Callable
'''
Assuming a Hamiltonian structure composed of two half-spaces joined at an interface,
this module constructs the full interfacial Green's function G(z, z'; ω) that satisfies the interface boundary conditions.
Hamiltonian looks like:
H = p^2/2m(z) + V(z) + V_int δ(z - z_int)
where m(z) and V(z) are piecewise constant in the left and right half-spaces, and V_int is the (constant) interface potential at z = z_int.
'''
class InterfacialGreensFunctionConstructor:
    def __init__(self,
                 greens_left_retard: Callable,
                 coincidence_left: sp.Matrix,
                 greens_right_retard: Callable,
                 coincidence_right: sp.Matrix,
                 interface_hamiltonian: sp.Matrix):
        self.G_L = greens_left_retard
        self.G_R = greens_right_retard
        self.coin_L = coincidence_left
        self.coin_R = coincidence_right
        self.H_int = interface_hamiltonian
        if self.coin_L.free_symbols:
            raise ValueError(f"'coincidence_left' must not have any free symbols. Got {self.coin_L.free_symbols}.")
        if self.coin_R.free_symbols:
            raise ValueError(f"'coincidence_right' must not have any free symbols. Got {self.coin_R.free_symbols}.")
        if self.H_int.free_symbols:
            raise ValueError(f"'interface_hamiltonian' must not have any free symbols. Got {self.H_int.free_symbols}.")

    # region Halfspace GF
    def halfspace_greens_function(self, side: str):
        if side == "left":
            G = self.Gl
        elif side == "right":
            G = self.Gr
        else:
            raise ValueError("side must be 'left' or 'right'")
        # HOW DO DEAL WITH THE BOUNDARY VALUE FOR EITHER SIDE AT THIS POINT???
        G00_inv = invert_matrix(self.get_boundary_value, symbolic=True)
        Gz0 = G.subs({self.z_prime: 0})
        G0zp = G.subs({self.z: 0})
        G_half = G - Gz0 @ G00_inv @ G0zp
        return G_half
    


    # endregion

    # region Full GF

    # endregion