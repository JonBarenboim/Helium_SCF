import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab
import units
from math import log
mlab.options.offscreen = True

plt.close('all')
mlab.close(all=True)

COLORMAP = 'OrRd'
A_COLORMAP = 'BuGn'
COLORMAP_LST = ['Reds', 'Blues', 'Greens', 'Purples']
INDEX_TO_QUANTNUMBER_MAP = ['1s', '2s', '2p', '3s']


def set_offscreen(flag):
    mlab.options.offscreen = flag


def plot_wavevector(wavefunc, plot_folder=None, plot_name_prefix=None):
    mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0,0,0), size=(800, 800))
    p = wavefunc.probdistr_unraveled()
    
    mlab.contour3d(p, contours=50, transparent=True, colormap=COLORMAP, opacity=0.4)
    cb = mlab.colorbar(nb_labels=2)
    cb.scalar_bar.unconstrained_font_size = True
    cb.label_text_property.font_size=16
    
    if plot_folder is not None:
        filename = f'{plot_folder}\\{plot_name_prefix}_wavefunction.png'
        mlab.savefig(filename)
    else:
        mlab.show()
    

'''
Plots the occupied orbitals in a WaveFunction
'''
def plot_wavefunction(wavefunc, grid, plot_folder, plot_name_prefix):
    X, Y, Z = grid.get_mesh()
    mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0,0,0), size=(800, 800))
    
    for (i, f) in enumerate(wavefunc.orbital_functions()):
        p = f(X, Y, Z)
        p2 = p.real**2 + p.imag**2
    
        mlab.contour3d(p2, contours=50, transparent=True, colormap=COLORMAP_LST[i], opacity=0.4)
        cb = mlab.colorbar(nb_labels=2, title=f"{INDEX_TO_QUANTNUMBER_MAP[i]} orbital")
        cb.scalar_bar.unconstrained_font_size = True
        cb.label_text_property.font_size=16
    
    if plot_folder is not None:
        filename = f'{plot_folder}\\{plot_name_prefix}_wavefunction.png'
        mlab.savefig(filename)
    else:
        mlab.show()


def plot_V(V, plot_folder, plot_name_prefix):
    V = np.array(V) # Required because V is not writable for some reason...
    V[V<=1e-5] = np.nan
    
    mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0,0,0), size=(800, 800))
    mlab.contour3d(V, contours=50, transparent=True, opacity=0.4, colormap=COLORMAP)
    cb = mlab.colorbar(nb_labels=2)
    cb.scalar_bar.unconstrained_font_size = True
    cb.label_text_property.font_size=16
   
    if plot_folder is not None:
        filename = f'{plot_folder}\\{plot_name_prefix}_V.png'
        mlab.savefig(filename)
    else:
        mlab.show()


def plot_energies(ms, energies, method, iterative_convergance=False, logarithmic=False):
    xlab = 'Iteration' if iterative_convergance else 'm'
    ylab = 'Energy (Hartrees)' if iterative_convergance else 'Energy (eV)'
    
    if not logarithmic:
        fig, ax = plt.subplots()
        ax.plot(ms, energies, marker='o')
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
    else:
        print(energies)

        l_energies = np.log10(-energies)
        l_ms = np.log10(ms)
        fig, ax=plt.subplots()
        ax.plot(l_ms, l_energies, marker='o')
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
    
        yt = ax.get_yticks()
        ax.set_yticklabels([ f'$10^{{ {e} }}$' for e in yt ])
        ax.set_xticks(l_ms)
        ax.set_xticklabels(ms)
    
    fig.suptitle(f'Energy for {method} Method ')
    
    return fig


def error_order(E, m, L):
    h = (2 * L)/(m + 1) 
    orders = [ log(E[i] / E[i+1]) / log(h[i] / h[i+1]) for i in range(len(E) - 1) ]
    return round(np.mean(orders), 4)


def plot_errors(ms, energies, expected_energy, method, expected_order = None, calc_order=False, iterative_convergance=False, L=None):
    xlab = 'Iteration' if iterative_convergance else 'm'
    errors = np.abs(energies - expected_energy)
    if calc_order: e_order = error_order(errors, ms, L)
    
    l_ms = np.log10(ms)
    l_errors = np.log10(errors)
    
    fig, ax = plt.subplots()
    ax.plot(l_ms, l_errors, marker='o')
    ax.set_xlabel(xlab)
    ax.set_ylabel('| Numerical - Expected |')
        
    if expected_order:
        ax.plot(l_ms, (l_ms - l_ms.min()) * -expected_order + l_errors.max() - 0.2, color='grey', linestyle='dashed')

    yt = ax.get_yticks()
    ax.set_yticklabels([ f'$10^{{ {e} }}$' for e in yt ])
    if calc_order: fig.text(0.6, 0.8, f'Error order = {e_order:.3f}')
    fig.suptitle(f'Error Behaviour for {method} Method')
    ax.set_xticks(l_ms)
    ax.set_xticklabels(ms)

    return fig

def plot_times(ms, times, method, average_iter_times = None):
    fig, ax1 = plt.subplots()
    ax1.plot(ms, times, marker='o')
    ax1.set_xlabel('m')
    ax1.set_ylabel('time (s)')
        
    if average_iter_times is not None:
        ax2 = ax1.twinx()
        ax2.plot(ms, average_iter_times, marker='o', color='r')
        ax2.set_ylabel('average iteration time (s)')
        ax2.yaxis.label.set_color('r')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    fig.suptitle(f'Wall Clock Time to Run {method} Method ')
    
    return fig
        
def plot_n_iters(ms, iters, method):
    fig, ax = plt.subplots()
    ax.plot(ms, iters, marker='o')
    ax.set_xlabel('m')
    ax.set_ylabel('# iteractions')
    fig.suptitle(f'Number of Iteractions Required for \n Convergance with {method} Method')
    return fig


def make_no_interaction_convergence_plots(ms, eigenfunctions, times, plot_folder, plot_name_prefix, expected_energy, L):
    # Plot energies and errors
    energies = np.fromiter([ units.atomic_to_ev(x.E) for x in eigenfunctions ], dtype=np.float)
    fig = plot_energies(ms, energies, 'No Interaction')
    if plot_folder is not None:
        filename = f'{plot_folder}\\{plot_name_prefix}_energy.png'
        fig.savefig(filename)
    else:
        fig.show()
    
    fig = plot_errors(ms, energies, expected_energy, 'No Interaction', expected_order=2, calc_order=True, L=L)
    if plot_folder is not None:
        filename = f'{plot_folder}\\{plot_name_prefix}_errors.png'
        fig.savefig(filename)
    else:
        fig.show()
    
    # Plot times
    fig = plot_times(ms, times, 'No Iteration')
    if plot_folder is not None:
        filename = f'{plot_folder}\\{plot_name_prefix}_runtime.png'
        fig.savefig(filename)
    else:
        fig.show()
    

def make_hartree_convergance_plots(ms, rets, plot_folder, plot_name_prefix, expected_energy):
    # Plot energies and errors
    energies = np.fromiter([ units.atomic_to_ev(r['eigenfunction'].E) for r in rets ], dtype=np.float)
    fig = plot_energies(ms, energies, 'Hartree')
    if plot_folder is not None:
        filename = f'{plot_folder}\\{plot_name_prefix}_energy.png'
        fig.savefig(filename)
    else:
        fig.show()
    
    fig = plot_errors(ms, energies, expected_energy, 'Hartree', expected_order=2, calc_order=False)
    if plot_folder is not None:
        filename = f'{plot_folder}\\{plot_name_prefix}_errors.png'
        fig.savefig(filename)
    else:
        fig.show()
    
    # Plot times
    times = np.fromiter([ r['total_time'] for r in rets ], dtype=np.float)
    avg_times = np.fromiter([ r['times'].mean() for r in rets ], dtype=np.float)
    fig = plot_times(ms, times, 'Hartree', avg_times)
    if plot_folder is not None:
        filename = f'{plot_folder}\\{plot_name_prefix}_runtime.png'
        fig.savefig(filename)
    else:
        fig.show()
    
    # Plot number of iterations needed
    iters = np.fromiter([ r['n_iters'] for r in rets ], dtype=np.float)
    fig = plot_n_iters(ms, iters, 'Hartree')
    if plot_folder is not None:
        filename = f'{plot_folder}\\{plot_name_prefix}_n_iters.png'
        fig.savefig(filename)
    else:
        fig.show()
    
    
def plot_iterative_method_convergance(ret, plot_folder, plot_name_prefix, method, expected_energy, skip=0):
    energies = np.array(ret['energies'])
    first_iter = 1 if len(energies) == ret['n_iters'] else 0
    iterations = np.arange(first_iter + skip, ret['n_iters']+1)
    energies = energies[skip:]
    
    fig = plot_energies(iterations, energies, method, iterative_convergance=True)
    if plot_folder is not None:
        plot_name = 'iterative_convergance'
        if skip: plot_name += f'_skip{skip}'
        filename=f'{plot_folder}\\{plot_name_prefix}_{plot_name}.png'
        fig.savefig(filename)
    else:
        plt.show(block=True)