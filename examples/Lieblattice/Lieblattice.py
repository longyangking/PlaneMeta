import sys
sys.path.append('../..')

import numpy as np
from PlaneMeta.tbmodel import Hamiltonian

import matplotlib.pyplot as plt  
from mpl_toolkits.mplot3d import Axes3D

if __name__=='__main__':
    H0 = np.array([ [0,0,0],
                    [0,0,0],
                    [0,0,0]])
    Vx = np.array([ [0,1,0],
                    [1,0,0],
                    [0,0,0]])
    Vy = np.array([ [0,0,0],
                    [0,0,1],
                    [0,1,0]])
    Vxy = np.zeros((3,3))
    Vyx = np.zeros((3,3))
    
    model = Hamiltonian(H0=H0, Vs=(Vx,Vy,Vxy,Vyx))
    kpoints = 25
    kxs, kys, bulkbands = model.bulkband(kspan=[[0,np.pi],[0,np.pi]],kpoints=kpoints, is_sort=True)
    
    plt.figure()
    for i in range(np.size(bulkbands,axis=1)):
        plt.plot(kxs,np.real(bulkbands[:,i]))

    kxs, kys, bulkbands = model.allband(kpoints=(25,25),kspan=((0,np.pi),(0,np.pi)),is_sort=True)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X,Y = np.meshgrid(kxs,kys)
    
    for i in range(np.size(bulkbands,axis=2)):
        ax.plot_surface(X, Y, np.real(bulkbands[:,:,i]), cmap=plt.get_cmap('rainbow'))
    
    plt.show()

