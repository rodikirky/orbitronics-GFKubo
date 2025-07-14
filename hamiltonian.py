import numpy as np
basis_default = np.array([1,0,0],[0,1,0],[0,0,1])

def potential_magnetic(MASS: float, ORBITAL_TEXTURE_COUPLING: float, 
                            EXCHANGE_COUPLING: float, MAGNETISATION: np.ndarray, 
                            momentum: np.ndarray, basis: np.ndarray = basis_default):
  """
  Computes the potential energy for a ferromagnetic or non-ferromagnetic system, 
  depending on the parameters provided, especially the coupling strength of the 
  exchange interaction.
  
  Parameters:
  MASS (float): Mass of the particle/object.
  ORBITAL_TEXTURE_COUPLING (float): Coupling constant related to orbital texture.
  EXCHANGE_COUPLING (float): Exchange coupling constant.
  MAGNETISATION (np.ndarray): Magnetisation vector.
  momentum (np.ndarray): Momentum vector.
  basis (np.ndarray): Basis vectors for the potential calculation.
  
  Returns:
  np.ndarray: The potential energy matrix.
  """
  # Identifying the parameters and variables
  m = MASS
  gamma = ORBITAL_TEXTURE_COUPLING
  J = EXCHANGE_COUPLING
  M = MAGNETISATION
  k = momentum # This is not a constant, we integrate over this variable later
  
  # Constructing the angular momentum vector based on the 
  L_x = np.array([0, 0, 0], [0, 0, -1j], [0, 1j, 0])
  L_y = np.array([0, 0, 1j], [0, 0, 0], [-1j, 0, 0])
  L_z = np.array([0, -1j, 0], [1j, 0, 0], [0, 0, 0])
  if basis == basis_default:
    L = np.array([L_x, L_y, L_z])
  else:
    U = basis
    L = np.array([U @ L_x @ U.T, U @ L_y @ U.T, U @ L_z @ U.T])
  
  # Putting together the potential energy
  V = np.dot(k, k) / (2 * m) + gamma * (np.dot(k, L))**2 + J * np.dot(M, L)

  return V

def building_hamiltonian(MASS: float, potential: np.ndarray, momentum: np.ndarray):
  """
  Constructs the Hamiltonian matrix for a given mass and potential.
  
  Parameters:
  mass (float): Mass of the particle/object.
  momentum (np.ndarray): Momentum vector.
  potential (np.ndarray): Potential energy matrix.
  
  Returns:
  np.ndarray: The Hamiltonian matrix.
  """
  # Identifying the parameters and variables
  m = MASS
  V = potential
  k = momentum

  # Constructing the Hamiltonian matrix
  H = np.dot(k, k) / (2 * m) + V
  
  return H
