import numpy as np
import STOnGbasis
from wavefunction import WaveFunction
import time

    
'''
Roothaan method with a basis of m orbitals constructed from n contracted Gaussians
Likely only works with n=3, m=2
Returns a WaveFunction
stop after *TWO* iterations with delta<convergance
'''
def roothaan(N, n, m, Z, convergance=1e-4, max_iter=50, verbose=False, return_progress=False):
    function_start = time.process_time()
    
    if return_progress:
        times = []
        energies = []
        Evecs = []
        Cs = []
        deltaPs = []
        deltaEs = []
        wavefunctions = []
    
    basis = STOnGbasis.STOnGBasis(n, m)
    Hcore = basis.get_Hcore(Z)
        
    # Matrix of coefficients
    C = np.zeros((m, m))
    # initial guess: P=0, ie F = Hcore
    P = np.zeros((m, m))    
    
    Evec = np.zeros(m)
    E = 0
    delta = np.inf
    delta2 = np.inf
    deltaE = np.inf
    iteration = 0
    
    while iteration < max_iter and (delta > convergance or delta2 > convergance):
        iteration += 1
        iter_start_t = time.process_time()        
        
        # Form Fock matrix
        G = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                for k in range(m):
                    for l in range(m):
                        G[i, j] += P[k, l] * basis.TEE[i, j, k, l]
        F = Hcore + G
        
        # Transform problem F C= S C E to eigenvalue problem Fprime Cprime = Cprime E
        Fprime = basis.X.dot(F.dot(basis.Xi))
        
        # Solve
        Evec_new, Cprime = np.linalg.eig(Fprime)
        
        # Revert back to original basis
        C = basis.Xi.dot(Cprime)
        
        wavefunc = WaveFunction(N, basis, C, Evec_new, Z)
        E_new = wavefunc.E
        
        # Form new density matrix
        P_new = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                for k in range(m):
                    P_new[i, j] = 2 * C[i, k] * C[j, k]
        
        # Calculate deltas
        delta2 = delta
        delta = np.linalg.norm(P_new - P, np.inf)
        deltaE = abs(E_new - E)
        
        # Update
        Evec = Evec_new
        E = E_new
        P = P_new
        
        t = time.process_time() - iter_start_t
        
        if return_progress:
            times.append(t)
            energies.append(E)
            Evecs.append(Evec)
            Cs.append(Cs)
            deltaPs.append(delta)
            deltaEs.append(deltaE)
            wavefunctions.append(wavefunc)
            
        if verbose:
            # TODO: move to a separate module
            def pretty_print_array(a):
                s = a.shape
                if len(s) == 2:
                    make_str = lambda row: ", ".join([ f"{i:.4f}" for i in a[row, :] ])
                    strs =  [ "[" + ", " + make_str(row) + "]" for row in range(s[0]) ]
                    return "[" + ", ".join(strs) + "]"
                if len(s) == 1:
                    return "[" +  ", ".join([ f"{i:.4f}" for i in a ]) + "]"
            print(f"iteration {iteration} \t E = {E:.4f} \t deltaP = {delta:.8f} \t deltaE = {deltaE:.8f} \t time={t}")
            print(f"\t\t P = {pretty_print_array(P)}")
            print(f"\t\t E = {pretty_print_array(Evec)}")
            print(f"\t\t C = {pretty_print_array(C)}")
            print()
            
    t_total = time.process_time() - function_start
    if verbose:
        print(f"Total function time: {t_total}")
    
    if return_progress:
        res = {'eigenfunction': wavefunc, 'wavefunctions': wavefunctions,
               'n_iters':iteration, 'times':times, 'energies': energies, 'Evecs': Evecs, 
               'deltaPs': deltaPs, 'deltaEs': deltaEs, 'total_time':  t_total}
    else:
        res =  wavefunc
        
    return res
        