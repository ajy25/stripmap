import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry.polygon import Polygon as ShapelyPoly, orient
from shapely.geometry.polygon import LinearRing as ShapelyRing

from num_methods import stparam, stmap, stinvmap

class Polygon:

    w = None
    n = None
    alpha = None
    shapely_poly = None

    def __init__(self, x_vertices: np.array, y_vertices: np.array) -> None:
        '''Constructs a Polygon object (Gamma) with complex vertices w.

        Parameters:
            - x_vertices: 1d array of real numbers representing x coordinates
            - y_vertices: 1d array of real numbers representing y coordinates
        
        Notes: 
            - The Polygon is assumed to be bounded. The input vertices must
            be given counterclockwise.
        '''

        print('\npolygon initialization attempted')

        # check if input is valid
        if len(x_vertices) != len(y_vertices):
            raise Exception('invalid input vertices')

        vect = []

        for i in range(len(x_vertices)):
            tup = (x_vertices[i], y_vertices[i])
            vect.append(tup)

        ring = ShapelyRing(vect)
        if not ring.is_simple:
            print('WARNING: linear ring (polygon bound) is not valid')
            print('attempting to address invalid ring')
            vect = self.fix_ring(vect)
        else:
            print('polygon appears valid')

        self.shapely_poly = ShapelyPoly(vect)
        self.shapely_poly = orient(self.shapely_poly, sign=1)
        x_vertices, y_vertices = self.shapely_poly.exterior.coords.xy
        x_vertices = x_vertices.tolist()
        y_vertices = y_vertices.tolist()
        x_vertices.pop()
        y_vertices.pop()

        # combine for complex w, determine n
        self.w = np.array(x_vertices) + 1j * np.array(y_vertices)
        print(self.w)
        self.n = len(self.w)

        # compute alpha
        self.compute_angles()

        print('polygon initialization successful')
        print('_________________________________\n')

    def compute_angles(self) -> None:
        '''Determines the interior angles alpha from the vertices w.'''

        # if angles have already been assigned, do nothing
        if self.alpha != None:
            return
        
        # if no vertices, set alpha to be empty
        if self.n == 0:
            self.alpha = np.array([])
            return
        
        # compute angles
        incoming = self.w - np.roll(self.w, 1)
        outgoing = np.roll(incoming, -1)
        self.alpha = np.mod(np.angle(np.multiply(-incoming, \
            np.conjugate(outgoing))) / np.pi, 2)

        # check angles sum to appropriate value
        angles_tol = 1.0 * 10 ** (-5)
        if np.abs(np.sum(self.alpha) - (self.n - 2)) > angles_tol:
            print('Invalid polygon. Angles sum to ' + \
                str(np.sum(self.alpha)) + '; expected angle sum is ' + \
                str(self.n - 2))
    
    def plot_poly(self) -> tuple:
        '''Returns a matplotlib plot depicting the polygon.'''

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        # to_plot = self.w
        to_plot = np.hstack([self.w, [self.w[0]]])
        ax.plot(np.real(to_plot), np.imag(to_plot))
        return fig, ax
    
    def is_in_poly(self, test_x: float, test_y: float) -> bool:
        '''Return true if the point (test_x, test_y) is within the polygon.'''

        point = ShapelyPoint(test_x, test_y)
        return self.shapely_poly.contains(point)
    
    def fix_ring(self, vect: list) -> list:
        '''Attempts to fix linear ring.'''
        
        for i in range(len(vect)):
            vect_copy = copy.deepcopy(vect)
            temp = vect_copy.pop(i)

            ring = ShapelyRing(vect_copy)
            
            if ring.is_valid:

                for j in range(len(vect_copy)):
                    
                    vect_copy_copy = copy.deepcopy(vect_copy)
                    vect_copy_copy.insert(j, temp)
                    test_ring = ShapelyRing(vect_copy_copy)

                    if test_ring.is_valid:
                        return vect_copy_copy
        
        print('WARNING: linear ring fix attempt unsuccessful')
        return vect

    
    # getters
    def get_vertices(self) -> np.array:
        '''Returns a shallow copy of the vertices of the polygon.'''
        return copy.copy(self.w)
    
    def get_angles(self) -> np.array:
        '''Returns a shallow copy of the interior angles of the polygon.'''
        return copy.copy(self.alpha)

    def get_size(self) -> int:
        '''Returns the size of the polygon.'''
        return self.n

class Stripmap:

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
            - p: a Polygon object
            - ends: a two-element array representing the ends of the strip
        
        Note:
            - The Polygon is assumed to have at least 3 vertices, none of which 
            are infinite.
        '''

        print('\nmap initialization attempted')

        # correct for zero-indexing; we assume input is one-indexed
        # ends = np.add(ends, -1)

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

        print('\nmap initialization successful with following parameters:')
        print(self)
        print('_______________________________________________________\n')

        return
    
    def plot_poly(self) -> None:
        '''Visualization of Polygon vertices and the stripmap.'''

        fig, ax = self.p.plot_poly()
        ax.plot(np.real(self.p.w[self.ends[0]]), 
                np.imag(self.p.w[self.ends[0]]), 'ro')
        ax.plot(np.real(self.p.w[self.ends[1]]), 
                np.imag(self.p.w[self.ends[1]]), 'ro')
        plt.show()
        return fig, ax

    def eval(self, xp: np.array, yp: np.array) -> tuple:
        '''Evaluates the forward map at points wp in the polygon.
        
        Parameters:
            - xp: vector of x coords within the polygon whose inverse mapping is 
                to be evaluated
            - yp: vector of y coords within the polygon whose inverse mapping is 
                to be evaluated
        
        Returns:
            - wp: invmapped points from within the polygon to the infinite strip
        '''

        print('\neval attempted')

        eps = .2204 * 10 ** (-16)

        if len(xp) != len(yp):
            raise Exception('Invalid input: number of x coordinates does' + \
                ' not match number of y coordinates.')

        for i in range(len(yp)):
            if yp[i] > 1 + eps or yp[i] < 0 - eps:
                raise Warning(str('The y-coord ' + str(yp[i]) + \
                    ' is not within the strip.'))

        zp = np.array(xp) + 1j * np.array(yp)
        self.zp = zp

        wp = stmap(zp, self)
        self.wp = wp

        print('eval successful')
        print('_______________\n')

        return np.real(wp), np.imag(wp)
    
    def evalinv(self, xp: np.array, yp: np.array) -> tuple:
        '''Evaluates inverse of the map at points wp in the polygon.
        
        Parameters:
            - xp: vector of x coords within the polygon whose inverse mapping is 
                to be evaluated
            - yp: vector of y coords within the polygon whose inverse mapping is 
                to be evaluated
        
        Returns:
            - zp: invmapped points from within the polygon to the infinite strip
        '''
        print('\nevalinv attempted')

        zp = np.empty(len(xp), dtype='complex_')

        zp[:] = np.NaN

        idx = []
        idx = []

        for i in range(len(xp)):
            if self.p.is_in_poly(xp[i], yp[i]):
                
                idx.append(i)

            else:
                print(str('The point (' + str(xp[i]) + \
                    ", " + str(yp[i]) + ') is not within the polygon.'))

        wp = np.array(xp) + 1j * np.array(yp)
        self.wp = wp

        zp[idx] = stinvmap(wp[idx], self)
        self.zp = zp

        print('evalinv successful')
        print('__________________\n')
        
        return np.real(zp), np.imag(zp)
    
    def get_info(self) -> str:
        '''Returns information regarding the map as a string.'''

        out = str(self.info) + "\nc:" + str(self.c)

        return out
    
    def get_w(self) :
        '''Returns a copy of vertices array.'''
        return copy.copy(self.p.get_vertices())
    
    def get_z(self):
        '''Returns a copy of prevertices array.'''
        return copy.copy(self.z)
    
    def get_beta(self):
        '''Returns a copy of beta array.'''
        return copy.copy(self.beta)

    def get_qdata(self):
        '''Returns a copy of qdata array.'''
        return copy.copy(self.qdata)
    
    def get_ends(self):
        '''Returns a copy of ends array.'''
        return copy.copy(self.ends)

    def __str__(self):
        '''Returns a string representation of the map.'''
        return self.get_info()

    

