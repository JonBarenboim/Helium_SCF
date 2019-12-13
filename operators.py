import numpy as np
from scipy import sparse
from random import randint

'''
The hamiltonian for the no-interaction approximation
'''
def H_no_interaction(grid, Z, order):
    L = laplacian(grid, order)
    A = VN(grid, Z, order)
    H = - L/2 - A
    return H

'''
The hamiltonians for the hartree method
Returned as a two-tuple
'''
def H_hartree(grid, order, Z, psi1, psi2):
    h0 = H_no_interaction(grid, Z, order)
    V1 = V_hartree(grid, order, psi2)
    V2 = V_hartree(grid, order, psi1)    
    return (h0 + V1, h0 + V2)


'''
Returns the Laplacian operator for a 3D grid of m interior points in each direction
'''
def laplacian(grid, order):
    T = FD2(grid, order)
    I = np.identity(grid.m) 
    L = sparse.kron(I, sparse.kron(I, T)) + sparse.kron(I, sparse.kron(T, I)) + sparse.kron(T, sparse.kron(I, I))
    return L

'''
Approximate the electron-nucleus interaction 
Vn = Z / r
'''
def VN(grid, Z, order):
    r = lambda x, y, z: np.sqrt(x**2 + y**2 + z**2)
    R = r(*grid.get_mesh())
    R = sparse.diags(R.flatten(order='C'), format='csr')
    np.reciprocal(R.data, out=R.data)
    return Z * R
  

'''
Approximate the electron-electron interaction for the hartree method
V_eff_1(r; psi_2) = \int |psi(r2)|^2 / |r - r_2| dr2
'''
def V_hartree(grid, order, psi2):
    # For each point approximate V_eff_1(x1, y1, z1) with Monte-Carlo integration
    # TODO: analysis of how many points needed
    N = 1000
    
    '''
    Evaluate the function V(r1; psi2) at the point (x1, y1, z1)
    V = int |psi2(r2)|^2 / r12 dr2
    for each point (x1, y1, z1) approximate V by the monte-carlo method with 
    f = |psi2(r2)|^2 / |r2 - r1| and N random points (x2, y2, z2)
    '''
    def V(x1, y1, z1):
        # evaluate the integrand at a point x2, y2, z2 given by indeces i2, j2, k2
        # i2, j2, k2 are in 1..grid.m
        def f(i2, j2, k2):
            x2, y2, z2 = grid.get_xyz(i2, j2, k2)
            r12 = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
            p = psi2[grid.raveled_index(i2, j2, k2)]
            p2 = p.real**2 + p.imag**2
            
            # Since V is evaluated on the entire grid at once, and the randomly sampled point is 
            # also on the grid, one of the evaluations gives a zero distance r12. 
            # Avoid this issue by finding that point and for that point only evaluating the integrand 
            # at another point
            # TODO: there has to be a better way to do this
            
            # Get index of the bad point
            b_rav_ind = grid.raveled_index(i2, j2, k2)
            
            # For now set r12 at the point to 1 so the division doesn't fail, and fix later
            r12[b_rav_ind] = 1
            # and now we can safely calculate f
            ret = p2 / r12
            
            # Get a new value for f
            # find a new point still on the interior of the grid
            a_i2 = 2 if i2 == 1 else i2-1 
            a_j2 = 2 if j2 == 1 else j2-1
            a_k2 = 2 if k2 == 1 else k2-1
            # coordinates of the new point
            a_x2, a_y2, a_z2 = grid.get_xyz(a_i2, a_j2, a_k2)
            # coordinates of just the bad point
            a_x1 = x1[b_rav_ind]
            a_y1 = y1[b_rav_ind]
            a_z1 = z1[b_rav_ind]
            a_r12 = np.sqrt((a_x2-a_x1)**2 + (a_y2-a_y1)**2 + (a_z2-a_z1)**2) 
            a_p = psi2[grid.raveled_index(a_i2, a_j2, a_k2)] # value of the wavefunction at the new point
            a_p2 = a_p.real**2 + a_p.imag**2
            ret[b_rav_ind] = a_p2 / a_r12       
            
            return ret
        
        '''
        Get a random point on the interior of the grid
        '''
        def sample():
            i = randint(1, grid.m)
            j = randint(1, grid.m)
            k = randint(1, grid.m)
            return (i, j, k)
        
        s = 0
        for n in range(N): s += f(*sample())
        
        vol = ((grid.L) * 2) ** 3
        return vol * s / N
    
    J = V(*grid.get_raveled_mesh())
    J = sparse.diags(J.flatten(order='C'), format='csr')
    return J
    

'''
Get the p-th order accurate centered finite difference 
approximation to the 2nd derivative with m points
'''
def FD2(grid, p):
    em = np.ones(grid.m)
    
    if p == 2:
        T = sparse.spdiags([em*-2, em, em], [0, -1, 1], grid.m, grid.m, format='csc')
    elif p == 4:
        T = sparse.spdiags([em*-30, em*16, em*16, em*-1, em*-1], [0, -1, 1, -2, 2], grid.m, grid.m, format='csc')
        T = T/12
    else:
        raise ValueError(f'order {p} not implemented in FD2')
        
    return T / grid.h**2
        