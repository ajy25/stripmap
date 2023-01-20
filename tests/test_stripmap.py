import pytest
import numpy as np
import scipy.linalg as la
import matlab
import matlab.engine
import sys, pathlib

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

if __name__ == '__main__':
    test_stderiv()
