import numpy as np
import Unitcell
import scipy.sparse.linalg as linalg

class Bandstructure:
    def __init__(self,ax,ay,epsilon,options=None):
        self.ax = ax
        self.ay = ay
        self.epsilon = epsilon
        self.unitcell = Unitcell.Unitcell(ax=ax,ay=ay,epsilon=epsilon)
        self.options = options
        
    def eigenvalue(self,kx,ky,bands=6):
        eigenValues,eigenVectors = self.eigensystem(kx,ky)
        return eigenValues[:bands]

    def eigenvector(self,kx,ky,bands=6):
        eigenValues,eigenVectors = self.eigensystem(kx,ky)
        return eigenVectors[:bands]

    def eigensystem(self,kx,ky,bands=6):
        '''
        The unit of eigenValues is w*a/2/pi/c
        '''
        hamiltonian = self.unitcell.Hamiltonian(kx,ky)
        eigenValues,eigenVectors = linalg.eigs(hamiltonian,k=bands,which='SR',tol=1.0*10**-12)
        eigenValues = np.sqrt(eigenValues)/2/np.pi
        return eigenValues[:bands],eigenVectors[:bands]

    def bulkband(self,kstart,kend,kpoints=15,bands=6):
        '''
        The unit of wave-vector is 1/a
        '''
        kxstart, kystart = kstart
        kxend, kyend = kend
        kx = np.linspace(kxstart,kxend,kpoints)
        ky = np.linspace(kystart,kyend,kpoints)

        bulkband = np.zeros((kpoints,bands),dtype=complex)
        for i in range(kpoints):
            bulkband[i] = self.eigenvalue(kx=kx[i],ky=ky[i],bands=bands)
        
        return kx,ky,bulkband