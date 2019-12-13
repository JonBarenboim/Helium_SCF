import pickle
from datetime import datetime
import time
import hartree
import roothaan
import plotting
import numpy as np
import os
import operators as ops
from grid import Grid

KNOWN_HE_GROUND_STATE_EV = -78.89
THEORY_NO_INT_GROUND_STATE_EV = -13.605 * 8

def no_interaction_convergence(ms, plot_folder, plot_name_prefix, save_to=None, L=5, Z=2, order=2):
    times = []
    eigenfunctions = []
    
    for m in ms:
        print('*'*25)
        print(f'\t\t m = {m}')
        print('*'*25)
        start = time.process_time()
        wavefunc = hartree.no_interaction_method(m, L, Z, order, how_many=1)[0]
        end = time.process_time()
        times.append(end - start)
        eigenfunctions.append(wavefunc)
    
    times = np.array(ms)
    eigenfunctions = np.array(eigenfunctions)
    
    plotting.make_no_interaction_convergence_plots(ms, eigenfunctions, times, plot_folder, plot_name_prefix, THEORY_NO_INT_GROUND_STATE_EV, L=L)
    
    if save_to:
        f = open(save_to, 'wb')
        pickle.dump({'ms': ms, 'eigenfunctions': eigenfunctions, 'times': times}, f)
    
    return (eigenfunctions, times)


def hartree_convergance(ms, plot_folder, plot_name_prefix, save_to=None, L=5, Z=2, order=2, verbose=True):
    rets = []
    
    for m in ms:
        print('*'*25)
        print(f'\t\t m = {m}')
        print('*'*25)
        ret = hartree.hartree_SCF(m, L, Z, order, return_progress=True, verbose=verbose, max_iter=50)
        rets.append(ret)
        
    rets = np.array(rets)
        
    plotting.make_hartree_convergance_plots(ms, rets, plot_folder, plot_name_prefix, KNOWN_HE_GROUND_STATE_EV)
        
    if save_to:
        f = open(save_to, 'wb')
        pickle.dump(rets, f)
    
    return rets


def run_hartree_fock(plot_folder, plot_name_prefix, Z=2, L=5, m=128, verbose=0):
    grid = Grid(m, L)
    wf = roothaan.roothaan(2, 3, 2, Z, convergance=1e-6, max_iter=50, verbose=True)
    plotting.plot_wavefunction(wf, grid, plot_folder, plot_name_prefix)



def plot_V(m, L, plot_folder, plot_name_prefix, Z=2):
    grid = Grid(m, L)
    noint = hartree.no_interaction_method(m, L, Z, 2)[0]
    psi2 = noint.u
    
    V = ops.V_hartree(grid, Z, psi2)
    V = np.diag(V.todense()).reshape((grid.m, grid.m, grid.m), order='C')
    plotting.plot_V(V, plot_folder, plot_name_prefix)
     


def analysis_pipeline():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = f"output\\{timestamp}"
    os.mkdir(outdir)
    
    ms = np.arange(2, 11) * 8
    plot_index = min(5, len(ms) - 1)
    
    no_int_ret = no_interaction_convergence(ms, outdir, 'no_interaction', save_to=f'{outdir}\\noint_output.obj', L=5, Z=2, order=2)
    hartree_ret = hartree_convergance(ms, outdir, 'hartree', save_to=f'{outdir}\\hartree_output.obj', L=5, Z=2, order=2)
    plotting.plot_wavevector(no_int_ret[0][plot_index], outdir, 'no_interaction')
    plotting.plot_wavevector(hartree_ret[plot_index]['eigenfunction'], outdir, 'hartree')
    plotting.plot_iterative_method_convergance(hartree_ret[plot_index], outdir, 'hartree', 'Hartree', KNOWN_HE_GROUND_STATE_EV)
    run_hartree_fock(outdir, 'roothaan', L=5, m=64)
    plot_V(36, 5, outdir, 'hartree')


analysis_pipeline()