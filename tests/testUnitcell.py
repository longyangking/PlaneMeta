import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
from Nezha.Unitcell import Unitcell


Nx = 3
Ny = 3
epsilon = np.ones((Nx,Ny))
unitcell = Unitcell(ax=1.0,ay=1.0,epsilon=epsilon)

A,DEX,DEY,DHX,DHY= unitcell.Hamiltonian(0,0)
print np.real(A.toarray())