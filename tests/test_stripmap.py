import pytest
import numpy as np
import scipy.linalg as la
import matlab
import matlab.engine
import sys, pathlib
import random
from polygon_sampling import generate_polygon

# dumb way of adding appropriate paths for testing
sys.path.append(str(pathlib.Path().resolve()) + '/')
sys.path.append(str(pathlib.Path().resolve()) + '/src/stripmap_ajy25')

from src.stripmap_ajy25.map import Polygon
from src.stripmap_ajy25.map import Stripmap
from src.stripmap_ajy25.num_methods import stpfun, stderiv


def test_stpfun():
    '''Simple test of stpfun, a cruicial function in solving
    the parameter problem.'''
    x_vert = np.array([0, 0.5, 1, 1.5, 2, 0, -1, -1.5, -2, -2])
    y_vert = np.array([2, 4, 6, 10, 12, 10, 8, 4, 1, 0])

    test_poly =  Polygon(x_vert, y_vert)

    test_map = Stripmap(test_poly, [1, 6])

    y1 = np.array([-0.031214892317576, 0.532128982172614, -2.758616578522062,
        1.671321856365236, 0.605751389975247, -0.221680763461464, 
        -5.516275062642263])
    n = 8
    nb = 4
    beta = test_map.get_beta()
    nmlen = np.array([-0.41573034-0.13483146j, 0.85459761+0.j, 0.43704832+0.j,
        0.85459761+0.j, 0.64477154+0.j, 0.21199958+0.j])
    left = np.array([0, 0, 1, 2, 4, 5, 6])
    right = np.array([7, 1, 2, 3, 5, 6, 7])
    cmplx = np.array([False, True, False, False, False, False, False])
    qdata = test_map.get_qdata()

    python_res = stpfun(y1, n, nb, beta, nmlen, left, right, cmplx, qdata)

    left = left + 1
    right = right + 1

    eng = matlab.engine.start_matlab()
    eng.cd('tests')
    y1 = matlab.double(y1.tolist(), is_complex=True)
    beta = matlab.double(beta.tolist(), is_complex=True)
    left = matlab.double(left.tolist(), is_complex=False)
    right = matlab.double(right.tolist(), is_complex=False)
    cmplx = matlab.logical(cmplx.tolist())
    qdata = matlab.double(qdata.tolist(), is_complex=True)
    nmlen = matlab.double(nmlen.tolist(), is_complex=True)

    n = matlab.double(int(n), is_complex=False)
    nb = matlab.double(int(nb), is_complex=False)

    result = eng.stpfun(y1,n,nb,beta,nmlen,left,right,cmplx,qdata)
    result = np.transpose(np.array(result))[0]
    print('matlab result: ', result)
    print('python result: ', python_res)
    dif_norm = la.norm(python_res - result)
    print('norm of difference: ', la.norm(python_res - result))

    assert np.abs(dif_norm) < 10 ** (-8)



def test_stderiv():
    '''Simple test of stderiv, a crucial function in both
    the process of solving for parameters and the inverse mapping.'''

    x_vert = np.array([0, 0.5, 1, 1.5, 2, 0, -1, -1.5, -2, -2])
    y_vert = np.array([2, 4, 6, 10, 12, 10, 8, 4, 1, 0])

    test_poly =  Polygon(x_vert, y_vert)

    test_map = Stripmap(test_poly, [1, 6])

    np.random.seed(1)

    # random points within a region of the polygon
    zp = 0.25 * np.random.rand(6) + 6j + 2j * np.random.rand(6)
    z = test_map.get_z()
    beta = test_map.get_beta()
    c = test_map.c

    result = stderiv(zp, z, beta, c)

    eng = matlab.engine.start_matlab()
    eng.cd('tests')

    zp = matlab.double(zp.tolist(), is_complex=True)
    z = matlab.double(z.tolist(), is_complex=True)
    beta = matlab.double(beta.tolist(), is_complex=False)
    c = matlab.double(c, is_complex=True)

    matlab_result = eng.stderiv(zp, z, beta, c)

    dif_norm = la.norm(matlab_result - result)

    assert np.abs(dif_norm) < 10 ** (-8)


def test_simple_polygons_params():
    '''Simple tests of the parameter problem on randomly generated n-gons. 
    In these tests, 6 <= n < 20. We assume that the ends are not too close to 
    one another; we therefore test with ends of [1, 4], [1, 5], ..., [1, n-2].
    '''
    random.seed(2)

    eng = matlab.engine.start_matlab()
    eng.cd('tests')

    for n in range(6, 20):

        out = generate_polygon(
            center=(0, 0),
            avg_radius=5,
            irregularity=0.8,
            spikiness=0.5,
            num_vertices=n)

        x = []
        y = []

        for tup in out:
            x.append(tup[0])
            y.append(tup[1])

        x_matlab = matlab.double(x, is_complex=False)
        y_matlab = matlab.double(y, is_complex=False)

        test_poly = Polygon(x, y)

        print('\nNew Polygon.')
        print(n)
        print(x)
        print(y)
        print('')

        for x in range(4, n-1):
            ends = [1, x]
            print('Ends: ', ends)
            ends_matlab = matlab.double(ends, is_complex=False)

            test_map = Stripmap(test_poly, ends)

            matlab_result = eng.paramstester(x_matlab, y_matlab, ends_matlab)
            matlab_z =  np.transpose(np.array(matlab_result[0]))[0]

            # norm of difference of zs
            result = test_map.get_z()
            result[np.isinf(result)] = 0

            matlab_z[np.isinf(matlab_z)] = 0

            dif_norm = la.norm(result - matlab_z)

            print(np.abs(dif_norm))
            assert np.abs(dif_norm) < 10 ** (-5)

if __name__ == '__main__':
    test_simple_polygons_params()
