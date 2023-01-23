import pytest
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from scipy.optimize import fsolve
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
from src.stripmap_ajy25.num_methods import stpfun, stderiv, scqdata, \
    iterate_solvers



eng = matlab.engine.start_matlab()
eng.cd('tests')

 

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

    left = left + 1
    right = right + 1

    y1_matlab = matlab.double(y1.tolist(), is_complex=True)
    beta_matlab = matlab.double(beta.tolist(), is_complex=True)
    left_matlab = matlab.double(left.tolist(), is_complex=False)
    right_matlab = matlab.double(right.tolist(), is_complex=False)
    cmplx_matlab = matlab.logical(cmplx.tolist())
    qdata_matlab = matlab.double(qdata.tolist(), is_complex=True)
    nmlen_matlab = matlab.double(nmlen.tolist(), is_complex=True)

    n_matlab = matlab.double(int(n), is_complex=False)
    nb_matlab = matlab.double(int(nb), is_complex=False)

    result = eng.stpfun(y1_matlab,n_matlab,nb_matlab,beta_matlab,nmlen_matlab,\
        left_matlab,right_matlab,cmplx_matlab,qdata_matlab)
    result = np.transpose(np.array(result))[0]
    print('matlab result: ', result)

    python_res = stpfun(y1, n, nb, beta, nmlen, left, right, cmplx, qdata)
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

        for j in range(4, n-1):
            ends = [1, j]
            print('\nNEXT ENDS: ' + str(ends) + '\t' + str(n) + '-gon:')
            print(x)
            print(y)
            ends_matlab = matlab.double(ends, is_complex=False)

            matlab_result = eng.paramstester(x_matlab, y_matlab, ends_matlab)
            matlab_z =  np.transpose(np.array(matlab_result[0]))[0]

            test_map = Stripmap(test_poly, ends)
            result = test_map.get_z()

            print('\nResults')
            print('matlab z: ', matlab_z)
            print('python z: ', result)
            # test_one_stparam(x, y, ends)
            

            # norm of difference of zs
            result[np.isinf(result)] = 0

            matlab_z[np.isinf(matlab_z)] = 0

            dif_norm = la.norm(result - matlab_z) / la.norm(matlab_z)

            print(np.abs(dif_norm))
            assert np.abs(dif_norm) < 10 ** (-5)

def get_stpfun_params(p: Polygon, ends):
    # define necessary local variables
    beta = p.get_angles() - 1              # betas
    N = len(beta)                       # number of vertices
    w = p.get_vertices()                     # vertices
    n = N - 2                           # number of prevertices

    map_tol = 10 ** (-8)                   # tolerance

    ends = np.array(ends) - 1

    renum = np.hstack((np.arange(ends[0], N), \
        np.arange(ends[0])))

    k = np.nonzero(renum == ends[1])[0][0] + 1
    
    # number of prevertices on the bottom edge of the strip
    nb = k - 2

    # renumber vertices to start with first element of ends
    w = w[renum]
    beta = beta[renum]

    # quadrature data
    nqpts = int(np.max([np.ceil(-np.log10(map_tol)), 4]))
    qdata = scqdata(beta, nqpts)

    atinf = beta <= -1

    # ignore images at ends of the strip
    w = np.delete(w, [0, k-1])
    atinf = np.delete(atinf, [0, k-1])

    # make an initial guess for z0
    z0 = np.zeros(n, dtype='complex_')

    if np.any(atinf):
        raise Exception('Polygon is assumed to be bounded.')

    scale = (np.abs(w[n-1] - w[0]) + np.abs(w[nb-1] - w[nb])) / 2

    z0[0:nb] = np.cumsum(np.abs(np.insert(w[1:nb] - w[0:(nb-1)], 0, 0))\
        / scale)
    
    if (nb + 1) == n:
        z0[n-1] = np.mean([z0[0], z0[nb-1]])
    else:
        z0[nb:n] = np.flip(np.cumsum(np.abs(np.insert(np.flip(\
            w[(nb+1):n]) - np.flip(w[nb:(n-1)]), 0, 0)) / scale))

    scale = np.sqrt(z0[nb-1] / z0[nb])

    z0[0:nb] = np.divide(z0[0:nb], scale)
    z0[nb:n] = 1j + np.multiply(z0[nb:n], scale)

    y0 = np.hstack([np.log(np.diff(z0[0:nb])), np.real(z0[nb]), \
        np.log(-np.diff(z0[nb:n]))])

    # find prevertices
    # left and right are zero-indexed
    left = np.add(np.hstack([np.array([1]), np.linspace(1, n-1, n-1)]), \
        -1).astype(int)
    right = np.add(np.hstack([np.array([n]), np.linspace(2, n, n-1)]), \
        -1).astype(int)
    left = np.delete(left, nb)
    right = np.delete(right, nb)

    # cmplx is a logical
    cmplx = np.add(right, -left) == 2
    cmplx[0] = False
    cmplx[1] = True
    nmlen = np.add(w[right], -w[left]) / (w[n-1] - w[0])
    nmlen[np.logical_not(cmplx)] = np.abs(nmlen[np.logical_not(cmplx)])
    nmlen = np.delete(nmlen, 0)

    return y0, n, nb, beta, nmlen, left, right, cmplx, qdata



def test_one_stpfun(x_vert, y_vert, ends, y, type='python'):

    p = Polygon(x_vert, y_vert)

    y = np.array(y)


    y0, n,nb,beta,nmlen,left,right,cmplx,qdata = get_stpfun_params(p, ends)
    # y = fsolve(stpfun, np.real(y0), (n, nb, beta, nmlen, left, \
    #         right, cmplx, qdata))

    if type == 'python':
        print('stpfun result: ', \
            stpfun(y, n,nb,beta,nmlen,left,right,cmplx,qdata))

    left = left + 1
    right = right + 1

    y = matlab.double(y.tolist(), is_complex=True)
    beta = matlab.double(beta.tolist(), is_complex=True)
    left = matlab.double(left.tolist(), is_complex=False)
    right = matlab.double(right.tolist(), is_complex=False)
    cmplx = matlab.logical(cmplx.tolist())
    qdata = matlab.double(qdata.tolist(), is_complex=True)
    nmlen = matlab.double(nmlen.tolist(), is_complex=True)
    n = matlab.double(int(n), is_complex=False)
    nb = matlab.double(int(nb), is_complex=False)

    if type == 'matlab':
        result = eng.stpfun(y,n,nb,beta,nmlen,left,right,cmplx,qdata)
        result = np.transpose(np.array(result))[0]
        print('stpfun result: ', result)


def test_one_stparam(x_vert, y_vert, ends):
    p = Polygon(x_vert, y_vert)

    y0, n,nb,beta,nmlen,left,right,cmplx,qdata = get_stpfun_params(p, ends)

    w_matlab = matlab.double(p.get_vertices().tolist(), is_complex=True)
    beta_matlab = matlab.double(beta.tolist(), is_complex=False)
    ends_matlab = matlab.double(ends, is_complex=False)

    y_matlab = np.transpose(eng.stparam_y(w_matlab, beta_matlab, ends_matlab))[0]

    try:
        y_python = fsolve(stpfun, np.real(y0), (n, nb, beta, nmlen, left, \
            right, cmplx, qdata), maxfev=100*(n+1), )
    except:
        print('\nWARNING: fsolve did not converge with default options.')
        print('Now iterating through root finding methods.\n')
        y_python = iterate_solvers(np.real(y0), n, nb, beta, nmlen, left, \
            right, cmplx, qdata, [])

    print('matlab stparam_y: ', y_matlab)
    print('python stparam_y: ', y_python)

    test_one_stpfun(x_vert, y_vert, ends, y_matlab, 'matlab')
    test_one_stpfun(x_vert, y_vert, ends, y_python, 'python')

def compute_root_error(zero):
    n = len(zero)
    compare = np.zeros(n)

    return la.norm(zero - compare) / la.norm(compare) 

if __name__ == '__main__':
    # test_one_stpfun(
    #     x_vert=[1.2507682826028579, 2.106123801943506, 0.0, 5.306749116636828, 0.4568448660952736, -0.42450069430131426, -9.940285128646508, -5.602465497232643],
    #     y_vert=[-4.133988646478907, -3.9396975775363514, 0.0, -0.7324962546473224, 0.38798328575872865, 0.5928871473628481, -1.0912064704761821, -3.175803772505633],
    #     ends=[1, 4],
    #     y=[0.9128, 3.5780, 0.2729, -0.1685, -4.8275]
    # )

    # test_one_stparam(
    #     x_vert=[7.55757022021153, 3.6037204096674387, -1.2080076076960657, -7.190979426784731, -1.3087735960750977, 1.8176840263958098, 3.193598439203918],
    #     y_vert=[1.7354629051920187, 4.128009422251863, 3.79281856434809, -0.6666587952340712, -7.257622034512454, -2.486510588547469, -0.8157083895934295],
    #     ends=[1, 4]
    # )

    test_simple_polygons_params()
    
