import numpy as np
from numpy import linalg

class Hamiltonian:
    def __init__(self, H0, Vs, origin=(0,0)):
        self.H0 = np.array(H0)
        self.Vx, self.Vy, self.Vxy, self.Vyx = Vs
        self.kx0, self.ky0 = origin
        self.nb_values = self.H0.shape[0]
        
    def fit(self):
        '''
        Fit the LCAO Hamiltonian based on the Band structure calculated by n        numerical results
        '''
        pass

    def eigenvalue(self,kx,ky):
        eigenValues,eigenVectors = self.eigensystem(kx,ky)
        return eigenValues

    def eigenvector(self,kx,ky):
        eigenValues,eigenVectors = self.eigensystem(kx,ky)
        return eigenVectors

    def eigensystem(self,kx,ky):
        '''
        The unit of eigenValues is w*a/2/pi/c
        '''
        hamiltonian = self.H0 + self.Vx*np.cos(kx + self.kx0) \
                        + self.Vy*np.cos(ky + self.ky0)  \
                        + self.Vxy*np.cos(kx + ky + self.kx0 + self.ky0) \
                        + self.Vyx*np.cos(kx - ky + self.kx0 - self.ky0)

        eigenValues,eigenVectors = linalg.eig(hamiltonian)
        #eigenValues = np.sqrt(eigenValues)/2/np.pi
        eigenValues = np.real(eigenValues)
        return eigenValues, eigenVectors

    def bulkband(self,kspan,kpoints=15,is_sort=False):
        kxspan, kyspan = kspan
        kxs = np.linspace(*kxspan,kpoints)
        kys = np.linspace(*kyspan,kpoints)

        bulkbands = np.zeros((kpoints,self.nb_values),dtype=complex)

        for i in range(kpoints): 
            eigenvalues = self.eigenvalue(kx=kxs[i],ky=kys[i])
            if is_sort:
                eigenvalues = np.sort(np.real(eigenvalues))
            bulkbands[i,:] = eigenvalues
        
        return kxs,kys,bulkbands

    def allband(self,kpoints=(15,15),kspan=((-np.pi,np.pi),(-np.pi,np.pi)),is_sort=False):
        nb_kx, nb_ky = kpoints
        kxspan, kyspan = kspan
        kxs = np.linspace(*kxspan,nb_kx)
        kys = np.linspace(*kyspan,nb_ky)

        bulkbands = np.zeros((nb_kx,nb_ky,self.nb_values),dtype=complex)
        for i in range(nb_kx):
            for j in range(nb_ky):
                eigenvalues = self.eigenvalue(kx=kxs[i],ky=kys[j])
                if is_sort:
                    eigenvalues = np.sort(np.real(eigenvalues))
                bulkbands[i,j,:] = eigenvalues
        
        return kxs,kys,bulkbands
        
