import numpy as np
import matplotlib.pyplot as plt

class Polygon:
    # fields
    w = None
    n = None
    alpha = None

    # methods
    def __init__(self, x_vertices: np.array, y_vertices: np.array) -> None:
        '''Constructs a Polygon object (Gamma) with complex vertices w.

        Parameters:
            x_vertices - 1d array of real numbers representing the x coordinates
            y_vertices - 1d array of real numbers representing the y coordinates
        
        Notes: 
            The Polygon is assumed to be bounded. The input vertices must
            be given counterclockwise.
        '''

        # check if input is valid
        if len(x_vertices) != len(y_vertices):
            raise Exception('Invalid input vertices.')
        
        # combine for complex w, determine n
        self.w = x_vertices + 1j * y_vertices
        self.n = len(self.w)

        # compute alpha
        self.compute_angles()

        # method end
        return
        
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
        if np.abs(np.sum(self.alpha) - (self.n - 2)) > np.finfo(float).eps:
            raise Exception('Invalid polygon. Angles sum to ', \
                np.sum(self.alpha) + ".")
        
        # method end
        return
    
    def plot_poly(self) -> plt.figure:
        fig = plt.figure()
        plt.plot(np.real(np.hstack([self.w, [self.w[0]]])), \
            np.imag(np.hstack([self.w, [self.w[0]]])))
        return fig
    
    # getters
    def get_vertices(self) -> np.array:
        return self.w
    
    def get_angles(self) -> np.array:
        return self.alpha

    def get_size(self) -> int:
        return self.n