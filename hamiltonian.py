import numpy as np
basis_default = np.array([1,0,0],[0,1,0],[0,0,1])
def hamiltonian(mass,gamma,exchange_coupling,basis):
  L=1
  M=1
  k=1
  m=mass
  g=gamma
  J=exchange_coupling
  h=k^2/(2*m)+g*(k*L)^2+J*M*L
  return h
  
