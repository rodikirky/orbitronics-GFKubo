from utils import invert_matrix
'''
Assuming a Hamiltonian structure composed of two half-spaces joined at an interface,
this module constructs the full interfacial Green's function G(z, z'; ω) that satisfies the interface boundary conditions.
Hamiltonian looks like:
H = p^2/2m(z) + V(z) + V_int δ(z - z_int)
where m(z) and V(z) are piecewise constant in the left and right half-spaces, and V_int is the (constant) interface potential at z = z_int.
'''
class InterfacialGreensFunctionConstructor:
    def __init__(self,
                 z,
                 z_prime,
                 greens_left,
                 greens_right,
                 potential_interface,
                 interface_position=0.0):
        self.z = z
        self.z_prime = z_prime
        self.Gl = greens_left
        self.Gr = greens_right
        self.V_int = potential_interface
        self.z_int = interface_position
        # no symbolic parameter; mode must be symbolic to substitute later

    def get_boundary_value(self):
        Gl = self.Gl
        Gr = self.Gr
        # Gl.subs({self.z: 0, self.z_prime: 0}) not allowed due to ambiguity at z=z'



    def get_halfspace_greens_function(self, side: str):
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
    

