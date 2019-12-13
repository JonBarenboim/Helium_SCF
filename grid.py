import numpy as np

class Grid(object):
    '''
    m is the number of interior points
    L is the domain of grid. -L <= x,y,z <= L
    '''
    def __init__(self, m, L):
        self.m = m
        self.L = L
        self.h = 2 * L / (m + 1)
        
    def get_ticks(self, include_boundary=False):
        ticks = np.linspace(-self.L, self.L, self.m+2)
        if include_boundary:
            return ticks
        return ticks[1:-1]
    
    def get_mesh(self, include_boundary=False):
        ticks = self.get_ticks(include_boundary)
        return np.meshgrid(ticks, ticks, ticks, indexing='ij')
    
    def get_raveled_mesh(self, include_boundary=False):
        X, Y, Z = self.get_mesh(include_boundary)
        return (X.flatten(order='C'), Y.flatten(order='C'), Z.flatten(order='C'))
    
    def index_to_coord(self, i):
        if not (isinstance(i, int) and i >=0 and i<= self.m + 1):
            raise ValueError("index outside of grid")
        return -self.L + i * self.h    
    
    def size(self, include_boundary=False):
        if include_boundary:
            return (self.m+2)**3
        return self.m**3
    
    '''
    Get the (x, y, z) coordinates of the point cooresponding to indeces (i, j, k)
    To be consistent with notation, i, j, k in 0..m+1, x_0 = -L, x_(m+1) = L
    '''
    def get_xyz(self, i, j, k):
        x = self.index_to_coord(i)
        y = self.index_to_coord(j)
        z = self.index_to_coord(k)
        return (x, y, z)
    
    ''' 
    Get the index in the raveled array corresponding to the index (i, j, k)
    the indeces (i, j, k) always refer to the index on the grid, not the index of the array
    while the output of this function is the index in the array.
    Use the `interior` keyword to control whether to get the array index in the
    interior-only or full aray
    '''
    # USE C-ORDERING EVERYWHERE!
    def raveled_index(self, i, j, k, include_boundary=False):
        if not include_boundary:
            return (k-1) + (j-1)*self.m + (i-1)*self.m**2
        else:
            return k + j*self.m + i*self.m**2

    def unraveled_index(self, i, j, k):
        pass