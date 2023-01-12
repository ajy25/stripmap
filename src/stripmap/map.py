import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
from polygon import Polygon
from num_methods.solve_param import *
from num_methods.compute_map import *

class Stripmap:
    '''
    STRIPMAP (Class)

    Description:

    Methods:

    Reference:
    '''

    p = None                # Polygon object: w (vertices), alpha (angles)
    z = None                # nparray, solved prevertices
    c = None                # float, parameter
    beta = None             # nparray, 1 - alpha
    ends = None             # nparray, indices (one-indexed) of strip ends
    qdata = None            # nparray for Gauss-Jacobi quadrature
    info = None             # pandas dataframe containing readable map info
    wp = None               # pre-invmapped points (within the polygon)
    zp = None               # invmapped points (on the strip)
    tol = 10 ** (-8)        # default tolerance


    def __init__(self, p: Polygon, ends: np.array) -> None:
        '''Constructs a Stripmap object and solves for its parameters.
        
        Parameters:
            p - a Polygon object
            index - a two-element array representing the ends of the strip
                    (one-indexed)
        
        Note:
            The Polygon is assumed to have at least 3 vertices, none of which 
            are infinite.
        '''

        # correct for zero-indexing; we assume input is one-indexed
        ends = np.add(ends, -1)

        # set up fields
        self.p = p
        self.beta = self.p.get_angles() - 1
        self.ends = ends

        # verify ends
        if len(ends) != 2:
            raise Exception('The input "ends" must only contain two values.')
        if ends[0] < 0 or ends[1] < 0 or ends[0] == ends[1]:
            raise Exception('The input "ends" is invalid.')
        
        # solve parameter problem
        self.z, self.c, self.qdata = stparam(self)

        # print out params
        info_dict = {'vertex': self.p.get_vertices(), 'prevertex': self.z, \
            'alpha': self.p.get_angles(), 'beta': self.beta}
        self.info = pd.DataFrame(data=info_dict)
        print(self.info)
        print("c:", self.c)
        
        # method end
        return
    
    def plot_poly(self) -> None:
        '''Visualization of Polygon vertices and the stripmap.'''

        fig = self.p.plot_poly()
    
    def evalinv(self, xp: np.array, yp: np.array) -> np.array:
        '''Evaluates inverse of the map at points wp in the polygon.
        
        Parameters:
            xp - vector of x coords within the polygon whose inverse mapping is 
                to be evaluated
            yp - vector of y coords within the polygon whose inverse mapping is 
                to be evaluated
        
        Returns:
            zp - invmapped points from within the polygon to the infinite strip
        '''

        wp = xp + 1j * yp
        self.wp = wp

        zp = stinvmap(wp)
        self.zp = zp
        
        return zp
    
    def get_info(self) -> None:
        '''Prints information regarding the map as an easily readable pandas 
        dataframe.
        '''

        print(self.info)
        print("c:", self.c)
    
    def get_w(self) :
        return copy.copy(self.p.get_vertices())
    
    def get_z(self):
        return copy.copy(self.z)
    
    def get_beta(self):
        return copy.copy(self.beta)

    def get_qdata(self):
        return copy.copy(self.qdata)
    
    def get_ends(self):
        return copy.copy(self.ends)

# if __name__ == '__main__':
#     x_vert = np.array([0, 0.5, 1, 1.5, 2, 0, -1, -1.5, -2, -2])
#     y_vert = np.array([2, 4, 6, 10, 12, 10, 8, 4, 1, 0])

#     test_poly =  Polygon(x_vert, y_vert)

#     test_map = Stripmap(test_poly, [1, 6])
