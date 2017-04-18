import sys
sys.path.append('../..')

import numpy as np
import matplotlib.pyplot as plt
from PlaneMeta.Bandstructure import Bandstructure

a = 1.0
r = 0.2*a
nx = 15
ny = nx

eps0 = 1.0
eps1 = 12.5
nmax = np.max([np.sqrt(eps0),np.sqrt(eps1)])

dx = a/nmax/nx
dy = a/nmax/ny

Nx = int(a/dx)+1
Ny = int(a/dy)+1

dx = 1.0/Nx
dy = 1.0/Ny

epsilon = np.ones((Nx,Ny))
for i in range(Nx):
   for j in range(Ny):
        if np.linalg.norm([(i+1)*dx-0.5,(j+1)*dy-0.5]) <= r:
            epsilon[i,j] = 12.25
        else:
            epsilon[i,j] = 1.0


fig1 = plt.figure()
plt.imshow(epsilon,interpolation='none')

bandstructure = Bandstructure(ax=a,ay=a,epsilon=epsilon)

kx,ky,bulkband =  bandstructure.bulkband([-np.pi,-np.pi],[np.pi/100,np.pi/100],bands=4)
kxs = kx
kys = ky
bulkbands = bulkband


kx,ky,bulkband =  bandstructure.bulkband([np.pi/100,0],[np.pi,0],bands=4)
kxs = np.concatenate((kxs,kx))
kys = np.concatenate((kys,ky))
bulkbands = np.concatenate((bulkbands,bulkband))

kx,ky,bulkband =  bandstructure.bulkband([np.pi,0],[np.pi,np.pi],bands=4)
kxs = np.concatenate((kxs,kx))
kys = np.concatenate((kys,ky))
bulkbands = np.concatenate((bulkbands,bulkband))

fig2 = plt.figure()
points = np.size(bulkbands,axis=0)
for i in range(np.size(bulkbands,axis=1)):
    plt.scatter(range(points),np.real(bulkbands[:,i]))
plt.title('Band Structure')
plt.xlabel('Wavevector')
plt.ylabel(r'Frequency $\omega a/2\pi c$')
plt.show()