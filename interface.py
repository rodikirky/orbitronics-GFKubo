from utils import invert_matrix
import sympy as sp
from typing import Callable, Union
'''
Assuming a Hamiltonian structure composed of two half-spaces joined at an interface,
this module constructs the full interfacial Green's function G(z, z'; ω) that satisfies the interface boundary conditions.
Hamiltonian looks like:
H = p^2/2m(z) + V(z) + V_int δ(z - z_int)
where m(z) and V(z) are piecewise constant in the left and right half-spaces, and V_int is the (constant) interface potential at z = z_int.
'''
class InterfacialGreensFunctionConstructor:
    # region Init
    def __init__(self,
                 greens_left_retard: Callable, # Functions of z and z'
                 mass_left: Union[float,sp.Float],
                 greens_right_retard: Callable,
                 mass_right: Union[float,sp.Float],
                 interface_hamiltonian: sp.Matrix # Evaluated 2D Hamiltonian of the interface
                 ):
        self.G_L = greens_left_retard
        self.m_L = sp.Float(mass_left)
        self.G_R = greens_right_retard
        self.m_R = sp.Float(mass_right)
        self.H_int = interface_hamiltonian
        if self.H_int.free_symbols:
            raise ValueError(f"'interface_hamiltonian' must not have any free symbols. Got {self.H_int.free_symbols}.")
    # endregion

    # region Halfspace GF
    def halfspace_greens_function(self, side: str) -> Callable:
        G = self._side_choice(side)
        G_coin = G(0,0)
        inv_G_coin = self._inverse(G_coin)

        # Callable half-space GF:
        def half_G(z,zp) -> sp.Matrix:
            theta_z = sp.Heaviside(-z)
            theta_zp = sp.Heaviside(-zp)
            G_half = theta_z * theta_zp * (G(z,zp) - (G(z,0) @ inv_G_coin @ G(0,zp)))
            return G_half
        
        return half_G  

    # endregion

    # region Full GF
    def full_interfacial_greens_function(self) -> Callable:
        def helper(z, side: str) -> sp.Matrix:
            G_side = self._side_choice(side)
            G_side_coin = G_side(0,0)
            inv_G_coin = self._inverse(G_side_coin)
            F_side = -G_side(z,0) @ inv_G_coin
            return F_side
        def helper_bar(zp, side: str) -> sp.Matrix:
            G_side = self._side_choice(side)
            G_side_coin = G_side(0,0)
            inv_G_coin = self._inverse(G_side_coin)
            F_side_bar = -inv_G_coin @ G_side(0,zp)
            return F_side_bar
        
        def G_full(z, zp):
            # Formula:
            # G(z,z') = G_L_bar(z,z') + G_R_bar(z,z') + F(z)G(0,0)F_bar(z')

            # Formula for F/F_bar:
            # F(z) = theta(-z)*F_L(z) + theta(z)*F_R(z)
            def F(r):
                F_L = helper(r, side="left")
                F_R = helper(r, side="right")
                F = sp.Heaviside(-r) * F_L + sp.Heaviside(r) * F_R
                return F
            def F_bar(rp):
                F_bar_L = helper_bar(rp, side="left")
                F_bar_R = helper_bar(rp, side="right")
                F_bar = sp.Heaviside(-rp) * F_bar_L + sp.Heaviside(rp) * F_bar_R

            # Formula for the boundary value:
            # G(0,0) = 1 / (-H_int + L_R - L_L), where L_side = 1/(2m_side) * F'(0)
            z_sym = sp.symbols("z", real=True, positive=True)
            F_z = F(z_sym)
            F_diff = sp.diff(F_z, z_sym)
            L_L = F_diff.subs({z_sym: 0}) / (2 * self.m_L)
            L_R = F_diff.subs({z_sym: 0}) / (2 * self.m_R)
            G00_inv = sp.Matrix(-self.H_int + L_R - L_L)
            G00 = self._inverse(G00_inv)

            # Constructing full GF:
            G_L_bar = self.halfspace_greens_function(side="left")
            G_R_bar = self.halfspace_greens_function(side="right")
            G_zzp = G_L_bar(z, zp) + G_R_bar(z, zp) + (F(z) @ G00 @ F_bar(zp))
            return G_zzp
        
        return G_full
        
    # endregion

    # region Internal utilities
    def _side_choice(self, side: str):
        if side == "left":
            G = self.G_L
        elif side == "right":
            G = self.G_R
        else:
            raise ValueError("side must be 'left' or 'right'")
        return G
    
    def _inverse(self, matrix) -> sp.Matrix:
        M = matrix
        det_M = M.berkowitz_det()
        adj_M = sp.Matrix(M.adjugate())
        inv_M = adj_M / det_M
        return inv_M
