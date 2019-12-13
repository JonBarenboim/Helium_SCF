import numpy as np

'''
Vector representation of a wavefunction
Holds only the interior points; it is assumed that the wavefunction is 0 on the boundary
'''
class WaveVector(object):
    # u is either a m x m x m array of u at the interior points
    # or a raveled version of that array with
    # u[i, j, k] = u(-L + (i+1)h, -L + (j+1)h, -L + (k+1)h)
    # stored in raveled version
    def __init__(self, u):
        s = u.shape
        if len(s) == 3 and s[0] == s[1] and s[1] == s[2]:
            self.m = int(s[0])
            self.u = np.flatten(u, order='C')
        else:
            self.m = int(np.cbrt(s[0]))
            self.u = u
        
    '''
    Return the wavefunction with the boundary points included
    '''
    def get_full_wavefunc(self):
        ufull = np.zeros((self.m + 2, self.m + 2, self.m + 2))
        ufull[1:-1, 1:-1, 1:-1] = np.reshape(self.u)
        return ufull
    
    def raveled(self):
        return np.ravel(self.u, order='C')

    def unraveled(self):
        return self.u.reshape((self.m, self.m, self.m), order='C')

    def probdistr(self):
        u = self.raveled()
        p = u.real**2 + u.imag**2
        return p / np.sum(p)
    
    def probdistr_unraveled(self):
        u = self.unraveled()
        p = u.real**2 + u.imag**2
        return p / np.sum(p)
    
    def norm(self):
        return np.linalg.norm(self.u)
    

'''
Store a solution to the two-electron Schrodinger equation
psi1 and psi2 are each a m^3 vector or m x m x m array of the wavefunction at the interior points
'''
class HartreeEigenfunction(WaveVector):
    def __init__(self, E1, E2, psi1, psi2, E):
        super().__init__(psi1 * psi2)
        self.E1 = E1
        self.E2 = E2
        self.E = E
        self.psi1 = psi1
        self.psi2 = psi2      
        
        
    def separate(self):
        wf1 = WaveVector(self.psi1)
        wf2 = WaveVector(self.psi2)
        return (wf1, wf2)
        
'''
Basis representation of a wavefunction
Holds the basis and the coefficients
'''
class WaveFunction():
    '''
    N is the number of electrons - the number of occupied orbitals is N//2. 
    N should always be even. This notation is used for consistency elsewhere
    basis holds the basis
    C is the coefficient matrix
    Evec is the vector of eigenvalues for each basis state
    E is the total energy
    Z is nucleus charge
    '''
    def __init__(self, N, basis, C, Evec, Z):
        self.C = C
        self.N = N
        self.basis = basis
        self.Evec = Evec
        self.Z = Z
        self.E = self._total_energy()
        
    '''
    A generator of functions representing the value of each occupied orbital 
    Each function takes a point (x, y, z)
    ''' 
    def orbital_functions(self):
        for i in range(self.N//2):
            def f(x, y, z):
                s = 0
                for j in range(self.basis.m):
                    s += self.C[i, j] * self.basis.basis_functions[j].f(x, y, z)
                return s
            yield f
    
    '''
    Calculate the total energy of the wavefunction
    '''
    def _total_energy(self):
        Hcore = self.basis.get_Hcore(self.Z)
        m = self.basis.m
        E = 0
        # note that we only add the energies for the first N//2 basis functions
        # since those are the occuied orbitals
        for a in range(self.N//2):
            ha = 0
            for i in range(m):
                for j in range(m):
                    ha += Hcore[i, j] * self.C[a, i] * self.C[a, j]
            E += 0.5 * (self.Evec[a] + ha)
        return E