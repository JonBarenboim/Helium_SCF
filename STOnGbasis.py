import numpy as np
import scipy.linalg as spl


'''
Primitive Gaussian function with parameter alpha
'''
class Gaussian():
    def __init__(self, alpha):
        self.alpha = alpha
        self.N = (2 * self.alpha / np.pi) ** (3/4)
    
    '''
    Give the value of the Gaussian wavefunction at coordinate x, y, z
    '''
    def f(self, x, y, z):
        rsq = x**2 + y**2 + z**2
        return self.N * np.exp(-self.alpha * rsq)

'''
Return (di, gi) for the slater-type m orbital approximated with n gaussians
m is the quantum number of the orbital (1 = 1s, 2 = 2s, 3 = 2p, etc)
gi are the primitive gaussians and di are the contraction coefficients
'''
def STOnG_coefficients(n, m):
    print(f"{n}   {m}")
    if n == 1 and m == 1:
        return ([1], [Gaussian(0.270950)])
    elif n == 2 and m == 1:
        di = [0.678914, 0.430129]
        gi = [Gaussian(0.678914), Gaussian(0.430129)]
        return (di, gi)
    elif n == 3 and m == 1:
        di = [0.444635, 0.535328, 0.154329]
        gi = [Gaussian(0.109818), Gaussian(0.405771), Gaussian(2.227660)]
        return (di, gi)
    elif n == 3 and m == 2:
        di = [0.700115, 0.399513, -0.099672]
        gi = [Gaussian(0.0751386), Gaussian(0.231031), Gaussian(0.994203)]
        return (di, gi)
    else:
        raise ValueError(f"Library does not contain coefficients for n={n}, m={m}") 


'''
The STO-nG basis function
'''
class STOngBasisFunction():
    
    '''    
    n is the number of Gaussians used in the contractions
    m is the quantum number of the orbital (1 = 1s, 2 = 2s, 3 = 2p, etc)
    '''
    def __init__(self, n, m):
        self.n = n
        self.m = m
        di, gi = STOnG_coefficients(n, m)
        self.di = di
        self.gi = gi
        
    ''' 
    Give the value of the basis wavefunction at coordinate x, y, z
    '''
    def f(self, x, y, z):
        s = 0
        for i in range(self.m):
            s += self.di[i] * self.gi[i].f(x, y, z)
        return s
        

'''
Holds the basis and constructs the one- and two-electron integrals
'''
class STOnGBasis():
    '''
    n is the number of Gaussians used in the contractions
    m is the size of the basis, ie the max quantum number
    '''
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.basis_functions = [ STOngBasisFunction(n, mm) for mm in range(1, m+1) ]
        self._initialize_matrices()
        
        
    '''
    Initialize all the matrices needed 
    T, V, S, TEE hold the integrals for each pair of basis functions
    X and Xi are the square root and inverse square root of S used to diagonalize the basis
    Builds everything except Hcore, which has an additional parameter
    '''
    def _initialize_matrices(self):
        self.T = np.zeros((self.m, self.m))
        for a in range(self.m):
            for b in range(self.m):
                self.T[a, b] = self.Tab(a, b)
                
        self.V = np.zeros((self.m, self.m))
        for a in range(self.m):
            for b in range(self.m):
                self.V[a, b] = self.Vab(a, b)
                
        self.S = np.zeros((self.m, self.m))
        for a in range(self.m):
            for b in range(self.m):
                self.S[a, b] = self.Sab(a, b)
                
        self.TEE = np.zeros((self.m, self.m, self.m, self.m))
        for a in range(self.m):
            for b in range(self.m):
                for c in range(self.m):
                    for d in range(self.m):
                        self.TEE[a, b, c, d] = self.tei(a, b, c, d)
        
        self.X = spl.sqrtm(self.S)
        self.Xi = np.linalg.inv(self.X)
                
            
    '''
    Return the Hcore matrix
    A separate method is used here since Hcore takes an additional parameter
    '''
    def get_Hcore(self, Z):
        return -self.T - Z * self.V

    ''' 
    Calculate the kinetic term for a pair of basis functions
    T(phi_a, phi_b) = \int phi_a(r1) 1/2 * nabla^2 phi_b(r1) dr1
    '''
    def Tab(self, a, b):
        phi1 = self.basis_functions[a]
        phi2 = self.basis_functions[b]
        
        s = 0
        for i in range(self.n):
            for j in range(self.n):
                d1i = phi1.di[i]
                d2j = phi2.di[j]
                Ni = phi1.gi[i].N
                Nj = phi2.gi[j].N
                alphai = phi1.gi[i].alpha
                alphaj = phi2.gi[j].alpha
                integral = - 3 * alphai * alphaj * (np.pi / (alphai + alphaj))**(3/2) / (alphai + alphaj)
                s += d1i * d2j * Ni * Nj * integral
        return s

    '''
    Calculate the electron-nucleus potential term for a pair of basis functions
    V(phi_a, phi_b) = \int phi_a(r1) Z/r phi_b (r1) dr1
    Calculated without the Z factor, which is added in later
    '''
    def Vab(self, a, b):
        phi1 = self.basis_functions[a]
        phi2 = self.basis_functions[b]
            
        s = 0
        for i in range(self.n):
            for j in range(self.n):
                d1i = phi1.di[i]
                d2j = phi2.di[j]
                Ni = phi1.gi[i].N
                Nj = phi2.gi[j].N
                alphai = phi1.gi[i].alpha
                alphaj = phi2.gi[j].alpha
                integral = 2 * np.pi / (alphai + alphaj)
                s += d1i * d2j * Ni * Nj * integral
        return s
    
    '''
    Calculate the overlap between a pair of basis functions
    S(phi_a, phi_b) = \int phi_a(r1) phi_b(r1) dr1
    '''
    def Sab(self, a, b):
        phi1 = self.basis_functions[a]
        phi2 = self.basis_functions[b]
        s = 0
        for i in range(self.n):
            for j in range(self.n):
                d1i = phi1.di[i]
                d2j = phi2.di[j]
                Ni = phi1.gi[i].N
                Nj = phi2.gi[j].N
                alphai = phi1.gi[i].alpha
                alphaj = phi2.gi[j].alpha
                integral = (np.pi / (alphai + alphaj))**(3/2)
                s += d1i * d2j * Ni * Nj * integral
        return s

    ''' 
    Calculate the two-electron integral for a set of basis functions
    [ab|cd] = \int \int phi_a(1) phi_b(1) phi_c(2) phi_d(2) / r12 dr1 dr2
    '''
    def tei(self, a, b, c, d):
        phi1 = self.basis_functions[a]
        phi2 = self.basis_functions[b]
        phi3 = self.basis_functions[c]
        phi4 = self.basis_functions[d]
        s = 0
        for i in range(self.n):
            for j in range(self.n):
                for k in range(self.n):
                    for l in range(self.n):
                        d1i = phi1.di[i]
                        d2j = phi2.di[j]
                        d3k = phi3.di[k]
                        d4l = phi4.di[l]
                        Ni = phi1.gi[i].N
                        Nj = phi2.gi[j].N
                        Nk = phi3.gi[k].N
                        Nl = phi4.gi[l].N
                        alphai = phi1.gi[i].alpha
                        alphaj = phi2.gi[j].alpha
                        alphak = phi3.gi[k].alpha
                        alphal = phi4.gi[k].alpha
                        integral = 2 * np.pi**(5/2) / ( (alphai + alphaj) * (alphak + alphal) * (alphai + alphaj + alphak + alphal)**(1/2) )
                        s += d1i * d2j * d3k * d4l * Ni * Nj * Nk * Nl * integral
        return s

