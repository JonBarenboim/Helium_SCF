from grid import Grid
from wavefunction import HartreeEigenfunction
import operators as ops
import scipy.sparse.linalg as splinalg
import numpy as np
import time
from random import randint
np.set_printoptions(linewidth=200)

def get_ground_eigenstate(H):
    eigvals, eigvecs = splinalg.eigsh(H, which='SA', k=1, tol=1e-5)
    return (eigvals[0], eigvecs[:,0])
    

'''
Calculate the extra term in the energy expression,
\iint  |psi1(1)|^1 |psi2(2)|^2 / r12 dr1 dr2
'''
def hartree_energy(psi1, psi2, grid):
    N = 1000
    def v(i1, j1, k1, i2, j2, k2):
        l1 = grid.raveled_index(i1, j1, k1)
        l2 = grid.raveled_index(i2, j2, k2)
        x1, y1, z1 = grid.get_xyz(i1, j1, k1)
        x2, y2, z2 = grid.get_xyz(i2, j2, k2)
        p1 = psi1[l1]
        p1 = p1.real**2 + p1.imag**2
        p2 = psi2[l2]
        p2 = p2.real**2 + p2.imag**2
        r12 = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
        return p1 * p2 / r12
    
    def sample():
        i1 = randint(1, grid.m)
        j1 = randint(1, grid.m)
        k1 = randint(1, grid.m)
        i2 = randint(1, grid.m)
        j2 = randint(1, grid.m)
        k2 = randint(1, grid.m)
        # Make sure we don't get the same point since the electron-electron potential diverges at r12=0
        if i1 == i2 and j1 == j2 and k1 == k2:
            i1 = 2 if i1 == 1 else i1-1
        return (i1, j1, k1, i2, j2, k2)
    
    s = 0
    for n in range(N): s += v(*sample())
    vol = ((grid.L) * 2) ** 3
    return vol * s / N
       
            
'''
Solve the Schrodinger equation for a two-electron atom with a no
    electron-electron interaction approximation
Returns a list of HartreeEigenfunctions ordered by energy
'''
def no_interaction_method(m, l, Z, order, how_many=1):
    grid = Grid(m, l)
    H = ops.H_no_interaction(grid, Z, order)
    eigenfunctions = []
            
    eigvals, eigvecs = splinalg.eigsh(H, which='SA', k=how_many, tol=1e-5)
    # eigvals, eigvecs are solutions for both phi1 and phi2
    # the overall wavefunction can be a Hartree function of any combination of these
    # build all of the possibilities
    for i in range(len(eigvals)):
        for j in range(len(eigvals)):
            E1 = eigvals[i]
            E2 = eigvals[j]
            phi1 = eigvecs[:, i]
            phi2 = eigvecs[:, j]
            eigenfunctions.append(HartreeEigenfunction(E1, E2, phi1, phi2, E1 + E2))
    # Sort by eigenvalue
    eigenfunctions.sort(key=lambda x: x.E)
        
    return eigenfunctions


'''
Solve the Schrodinger equation for a two-electron atom using the Hartree method
Returns the ground state HartreeEigenfunction
If verbose, print iteration, delta, and time    
Stop after *TWO* iterations with delta < convergance
if return_progress, returns (times, energies, deltas)
'''
def hartree_SCF(m, l, Z, order, convergance=1e-6, max_iter=100, verbose=False, return_progress=False):
    function_start = time.process_time()
        
    grid = Grid(m, l)
    
    # Get initial guess
    no_interaction_ground_state = no_interaction_method(m, l, 1, order, how_many=1)[0]
    psi1 = no_interaction_ground_state.psi1
    psi2 = no_interaction_ground_state.psi2
    E1 = no_interaction_ground_state.E1
    E2 = no_interaction_ground_state.E2
    E = E1 + E1 + hartree_energy(psi1, psi2, grid)
    
    iteration = 0
    delta = np.inf
    delta2 = np.inf
    
    # save progress
    if return_progress:
        times = []
        energies = []
        energies1 = []
        energies2 = []
        deltas = []
    
    if verbose:
        print(f"iteration 0 \t E = {E:.8f}")
    
    if return_progress:
        times.append(0)
        energies.append(E)
        energies1.append(E1)
        energies2.append(E2)
        deltas.append(delta)
        
    while iteration < max_iter and (delta > convergance or delta2 > convergance):
        iteration += 1
        if verbose:
            iter_start_t = time.process_time()
        
        H1, H2 = ops.H_hartree(grid, order, Z, psi1, psi2)
        E1_new, psi1_new = get_ground_eigenstate(H1)
        E2_new, psi2_new = get_ground_eigenstate(H2)
        
        delta_psi1 = np.linalg.norm(psi1_new - psi1, np.inf)
        delta_psi2 = np.linalg.norm(psi2_new - psi2, np.inf)
                
        E1 = E1_new
        E2 = E2_new
        psi1 = psi1_new
        psi2 = psi2_new        
        
        E_h = hartree_energy(psi1, psi2, grid)
        E_new = E1 + E1 + E_h
        
        delta2 = delta
        delta = abs(E_new - E)
        E = E_new
        
        if verbose:
            t = time.process_time() - iter_start_t
            print(f"iteration {iteration} \t E = {E:.6f} \t delta = {delta:.6f} \t time = {t}")
            print(f"\t\t E1 = {E1:.6f} \t E2 = {E2:.6f} \t E_h = {E_h:.6f}")
            print(f"\t\t delta_psi1 = {delta_psi1:.6f} \t delta_psi2 = {delta_psi2:.6f}")
        
        if return_progress:
            times.append(t)
            energies.append(E)
            energies1.append(E1)
            energies2.append(E2)
            deltas.append(delta)
            
    t_total = time.process_time() - function_start
    
    times = np.array(times)
    energies = np.array(energies)
    energies1 = np.array(energies1)
    energies2 = np.array(energies2)
    deltas = np.array(deltas)
            
    if verbose:
        print(f"Total function time: {t_total}")
    
    if return_progress:
        res = {'eigenfunction':  HartreeEigenfunction(E1, E2, psi1, psi2, E), 
               'n_iters':iteration, 'times':times, 'energies': energies, 'energies1': energies1, 
               'energies2': energies2, 'deltas': deltas, 'total_time':  t_total}
    else:
        res =  HartreeEigenfunction(E1, E2, psi1, psi2, E) 
        
    return res

