import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize, linewidth=500, precision=4)
import time
from grid import Grid
import hartree
import roothaan
import units
import plotting

KNOWN_HE_GROUND_STATE_EV = -78.89
THEORY_NO_INT_GROUND_STATE_EV = -13.605 * 8
   
plotting.set_offscreen(False)

def test_no_interaction_method(m, Z, L, order):    
    start = time.process_time()
    eigenfunctions = hartree.no_interaction_method(m, L, Z, order)
    print(f'Time to run: {time.process_time() - start}')
    print(len(eigenfunctions))
    
    for eigenfunction in eigenfunctions[0 : min(len(eigenfunctions), 5)]:
        print(units.atomic_to_ev(eigenfunction.E1))
        print(units.atomic_to_ev(eigenfunction.E2))
        print(units.atomic_to_ev(eigenfunction.E))
        print()
     
    plotting.plot_wavevector(eigenfunctions[0])
        
    
# Note: m has to be odd to avoid the origin where the nuclear potential diverges
def test_hartree_method(m, Z, L, order, verbose=True):
    start = time.process_time()
    ret = hartree.hartree_SCF(m, L, Z, order, max_iter=15, verbose=verbose, return_progress=True)
    eigenfunction = ret['eigenfunction']
    print(f'Time to run: {time.process_time() - start}')
    
    print(units.atomic_to_ev(eigenfunction.E1))
    print(units.atomic_to_ev(eigenfunction.E2))
    print(units.atomic_to_ev(eigenfunction.E))
    print()
    plotting.plot_wavevector(eigenfunction)
    plotting.plot_iterative_method_convergance(ret, None, 'hartree', 'Hartree', KNOWN_HE_GROUND_STATE_EV)
    plotting.plot_iterative_method_convergance(ret, None, 'hartree', 'Hartree', KNOWN_HE_GROUND_STATE_EV, skip=3)
    #plotting.plot_separate_wavefunctions2(eigenfunction, grid)
        
def test_roothaan_method(n, m, Z, grid, N=2):
    start = time.process_time()
    ret = roothaan.roothaan(N, n, m, Z, verbose=True, return_progress=True)
    wavefunc = ret['eigenfunction']
    print(f"Time to run: {time.process_time() - start}")
    
    print(units.atomic_to_ev(wavefunc.E))
    print(wavefunc.C)
    
    plotting.plot_wavefunction(wavefunc, grid, None, 'roothaan')
    plotting.plot_iterative_method_convergance(ret, None, 'roothaan', 'Roothaan', KNOWN_HE_GROUND_STATE_EV)
    

#test_no_interaction_method(24, 2, 5, 2)
#test_hartree_method(24, 2, 5, 2)
#test_roothaan_method(3, 2, 2, Grid(8, 5), N=2)
test_roothaan_method(3, 2, 2, Grid(64, 5), N=4)