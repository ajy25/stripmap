import copy
import numpy as np
import scipy.linalg as la
import random
from scipy.special import gamma
from scipy.optimize import fsolve, root
from scipy.integrate import solve_ivp

map_tol = 10 ** (-8)                    # default tolerance

def stparam(map, y_attempt = np.array([])) -> tuple:
    '''Attempts to solve for the parameters of the stripmap.'''

    # define necessary local variables
    beta = map.get_beta()               # betas
    N = len(beta)                       # number of vertices
    w = map.get_w()                     # vertices
    n = N - 2                           # number of prevertices
    ends = map.get_ends()               # ends
    map_tol = map.tol                   # tolerance

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

    if len(y_attempt) != 0:
        y0 = y_attempt
    else:
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

    # print('y0:', y0)

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


    y = iterate_solvers(y0, n, nb, beta, nmlen, left, \
        right, cmplx, qdata, skip=[])
    
    # print(y)

    z = np.zeros(n, dtype='complex_')
    z[1:nb] = np.cumsum(np.exp(y[0:(nb-1)]))
    z[nb:] = np.add(np.cumsum(np.hstack([np.array([y[nb-1]]), \
        -np.exp(y[nb:n])])), 1j)

    z = np.hstack([np.array([-np.inf]), z[0:nb], np.array([np.inf]), \
        z[nb:n]])
    
    mid = np.mean(z[1:3])

    g = stquad(np.array([z[2]]), np.array([mid]), np.array([2]), z, \
        beta, qdata) - stquad(np.array([z[1]]), np.array([mid]), \
        np.array([1]), z, beta, qdata)

    if len(g) != 1:
        raise ValueError('The length of g was not 1.')

    c = np.divide(w[0] - w[1], g[0])

    z[renum] = z
    qdata[:, renum] = qdata[:, np.arange(N)]
    qdata[:, N + 1 + renum] = qdata[:, N + 1 + np.arange(N)]

    # method return
    return z, c, qdata


def stpfun(y: np.array, n: int, nb: int, beta: np.array, \
    nmlen: np.array, left: np.array, right: np.array, cmplx: np.array, \
    qdata: np.array) -> np.array:
    '''Function that maps from vector to vector, used by fsolve.'''
    # print('')
    # print(y.tolist())
    # print('')

    z = np.zeros(n, dtype='complex_')
    z[1:nb] = np.cumsum(np.exp(y[0:(nb-1)]))

    z[nb:] = 1j + np.cumsum(np.concatenate(([y[nb-1]], \
        -np.exp(y[nb:(n-1)]))))

    zleft = z[left]
    zright = z[right]
    mid = np.mean(np.vstack([zleft, zright]), axis=0)

    # cmplx is logical; so is c2
    c2 = copy.copy(cmplx)
    c2[1] = False
    mid[c2] = mid[c2] - np.sign(left[c2]) * 1j / 2

    zs = np.hstack([np.array([-np.inf]), z[0:nb], \
        np.array([np.inf]), z[nb:n]])
    left = np.add(left, np.add(1, (left > nb-1).astype(int)))
    right = np.add(right, np.add(1, (right > nb-1).astype(int)))
    
    ints = np.zeros(n - 1, dtype='complex_')
    c2[0] = True
    id = np.logical_not(c2)

    ints[id] = stquadh(zleft[id], mid[id], left[id], zs, beta, \
        qdata) - stquadh(zright[id], mid[id], right[id], zs, \
        beta, qdata)

    z1 = np.add(np.real(zleft[c2]), 1j / 2)
    z2 = np.add(np.real(zright[c2]), 1j / 2)
    id = np.logical_not(id)

    ints[id] = stquad(zleft[id], z1, left[id], zs, beta, qdata)
    

    ints[id] = ints[id] + stquadh(z1, z2, np.zeros(len(z1)), zs, \
        beta, qdata)

    ints[id] = ints[id] - stquad(zright[id], z2, right[id], zs, \
        beta, qdata)

    absval = np.abs(ints[np.logical_not(cmplx)])

    if absval[0] == 0:
        rat1 = np.array([0])
        rat2 = np.array([0])
    else:
        if len(absval) == 2:
            rat1 = np.array([absval[1] / absval[0]])
        else:
            rat1 = np.divide(absval[1:], absval[0])

        if len(ints[cmplx]) == 1:
            rat2 = np.array(ints[cmplx] / ints[0])
        else:
            rat2 = np.divide(ints[cmplx], ints[0])

    rat_test = np.hstack([rat1, rat2])
    
    if 0 in rat_test or np.any(np.isinf(rat_test)) or \
        np.any(np.isnan(rat_test)):
        raise RuntimeWarning('WARNING: SEVERE CROWDING')
    
    cmplx2 = cmplx[1:]

    if len(rat1) != 0:
        F1 = np.log(np.divide(rat1, nmlen[np.logical_not(cmplx2)]))
    else:
        F1 = np.array([])
    
    if len(rat2) != 0:
        F2 = np.log(np.divide(rat2, nmlen[cmplx2]))
    else:
        F2 = np.array([])

    # print(np.real(np.hstack([F1, np.real(F2), np.imag(F2)])))
    # return np.real(np.hstack([F1, np.real(F2), np.imag(F2)]))
    return np.real(np.hstack([F1, np.real(F2), np.imag(F2)]))

def iterate_solvers(y0, n, nb, beta, nmlen, left, right, cmplx, qdata, \
    skip: list[str] = [], rec_depth: int = 0):

    print('begin param solve')
    print('recursion depth: ', rec_depth)

    if rec_depth == 20:
        return None
        

    # max_step = 1000 * max(la.norm(np.real(y0)), 1)
    xtol = 10 ** (-8) / 10

    try: 
        y, idct, ierf, mesg = fsolve(stpfun, np.real(y0), (n, nb, beta, nmlen, \
            left, right, cmplx, qdata), maxfev=100*(n+1), full_output=True)
        if ierf == 1:
            verify_root(y, n, nb, beta, nmlen, left, right, cmplx, \
                qdata)
            return y
    except:
        pass
    
    if 'fsolve_iter_factor' not in skip:
        # fsolve
        method = 'fsolve_iter_factor'
        factor = 0.1
        while factor <= 100:
            for i in range(5):
                randdiag = np.random.rand(n) * 2
                try:
                    y, idct, ier, mesg = \
                        fsolve(stpfun, np.real(y0), (n, nb, beta, nmlen, left, \
                        right, cmplx, qdata), maxfev=100*(n+1), factor=factor, \
                        xtol=xtol, full_output=True, diag=randdiag)
                    if ier == 1:
                        print('Method ' + method + ' was successful.')
                        if not verify_root(y, n, nb, beta, nmlen, left, right, \
                            cmplx, qdata):
                            skip.append(method)
                            rec_depth += 1
                            return iterate_solvers(y, n, nb, beta, nmlen, left,\
                                right, cmplx, qdata, skip, rec_depth)
                        else:
                            return y
                except:
                    pass
            factor = min(factor * 2, factor + 5)
        print('Method ' + method + ' unsuccessful.\n')

    if 'lm' not in skip:
        # lm
        method = 'lm'
        factor = 0.1
        while factor <= 100:
            try:
                options = {
                    'factor': factor,
                    'maxiter': 100 * (n+1),
                    'xtol': xtol
                }

                y = root(stpfun, np.real(y0), (n, nb, beta, nmlen, left, right, \
                    cmplx, qdata), method=method, options=options).x
                print('Method ' + method + ' was successful.')
                if not verify_root(y, n, nb, beta, nmlen, left, right, cmplx, \
                    qdata):
                    skip.append(method)
                    rec_depth += 1
                    return iterate_solvers(y, n, nb, beta, nmlen, left,\
                        right, cmplx, qdata, skip, rec_depth)
                else:
                    return y
            except:
                factor = min(factor * 2, factor + 5)
        print('Method ' + method + ' unsuccessful.\n')


    if 'broyden1' not in skip:
        # broyden1
        method = 'broyden1'
        factor = 0.1
        for linesearch in ['armijo', 'wolfe']:
            try:
                options = {
                    'maxiter': 10 * (n+1),
                    'xtol': xtol,
                    'ftol': xtol * 10,
                    'line_search': linesearch
                }

                y = root(stpfun, np.real(y0), (n, nb, beta, nmlen, left, right, \
                    cmplx, qdata), method=method, options=options).x
                print('Method ' + method + ' was successful.')
                if not verify_root(y, n, nb, beta, nmlen, left, right, cmplx, \
                    qdata):
                    skip.append(method)
                    rec_depth += 1
                    return iterate_solvers(y, n, nb, beta, nmlen, left,\
                        right, cmplx, qdata, skip, rec_depth)
                else:
                    return y
            except:
                pass
        print('Method ' + method + ' unsuccessful.\n')

    if 'broyden2' not in skip:
        # broyden2
        method = 'broyden2'
        factor = 0.1
        for linesearch in ['armijo', 'wolfe']:
            try:
                options = {
                    'maxiter': 10 * (n+1),
                    'xtol': xtol,
                    'ftol': xtol * 10,
                    'line_search': linesearch
                }

                y = root(stpfun, np.real(y0), (n, nb, beta, nmlen, left, right, \
                    cmplx, qdata), method=method, options=options).x
                print('Method ' + method + ' was successful.')
                if not verify_root(y, n, nb, beta, nmlen, left, right, cmplx, \
                    qdata):
                    skip.append(method)
                    rec_depth += 1
                    return iterate_solvers(y, n, nb, beta, nmlen, left,\
                        right, cmplx, qdata, skip, rec_depth)
                else:
                    return y
            except:
                pass
        print('Method ' + method + ' unsuccessful.\n')

    if 'df-sane' not in skip:
        # df-sane
        method = 'df-sane'
        factor = 0.1
        for linesearch in ['cruz', 'cheng']:
            try:
                options = {
                    'maxfev': 10 * (n+1),
                    'xtol': xtol,
                    'ftol': xtol * 10,
                    'line_search': linesearch
                }

                y = root(stpfun, np.real(y0), (n, nb, beta, nmlen, left, right, \
                    cmplx, qdata), method=method, options=options).x
                print('Method ' + method + ' was successful.')
                if not verify_root(y, n, nb, beta, nmlen, left, right, cmplx, \
                    qdata):
                    skip.append(method)
                    rec_depth += 1
                    return iterate_solvers(y, n, nb, beta, nmlen, left,\
                        right, cmplx, qdata, skip, rec_depth)
                else:
                    return y
            except:
                pass
        print('Method ' + method + ' unsuccessful.\n')

    if 'krylov' not in skip:
        # krylov
        method = 'krylov'
        factor = 0.1
        for linesearch in ['armijo', 'wolfe']:
            try:
                options = {
                    'maxiter': 10 * (n+1),
                    'xtol': xtol,
                    'ftol': xtol * 10,
                    'line_search': linesearch
                }

                y = root(stpfun, np.real(y0), (n, nb, beta, nmlen, left, right, \
                    cmplx, qdata), method=method, options=options).x
                print('Method ' + method + ' was successful.')
                if not verify_root(y, n, nb, beta, nmlen, left, right, cmplx, \
                    qdata):
                    skip.append(method)
                    rec_depth += 1
                    return iterate_solvers(y, n, nb, beta, nmlen, left,\
                        right, cmplx, qdata, skip, rec_depth)
                else:
                    return y
            except:
                pass
        print('Method ' + method + ' unsuccessful.\n')
    
    if 'excitingmixing' not in skip:
        # excitingmixing
        method = 'excitingmixing'
        factor = 0.1
        for linesearch in ['armijo', 'wolfe']:
            try:
                options = {
                    'maxiter': 10 * (n+1),
                    'xtol': xtol,
                    'ftol': xtol * 10,
                    'line_search': linesearch
                }

                y = root(stpfun, np.real(y0), (n, nb, beta, nmlen, left, right, \
                    cmplx, qdata), method=method, options=options).x
                print('Method ' + method + ' was successful.')
                if not verify_root(y, n, nb, beta, nmlen, left, right, cmplx, \
                    qdata):
                    skip.append(method)
                    rec_depth += 1
                    return iterate_solvers(y, n, nb, beta, nmlen, left,\
                        right, cmplx, qdata, skip, rec_depth)
                else:
                    return y
            except:
                pass
        print('Method ' + method + ' unsuccessful.\n')

    if 'hybr' not in skip:
        # hybr
        method = 'hybr'
        factor = 0.1
        while factor <= 100:
            try:
                options = {
                    'maxfev': 10 * (n+1),
                    'factor': factor,
                    'xtol': xtol
                }

                y = root(stpfun, np.real(y0), (n, nb, beta, nmlen, left, right, 
                cmplx, qdata), method=method, options=options).x
                print('Method ' + method + ' was successful.')
                if not verify_root(y, n, nb, beta, nmlen, left, right, cmplx, \
                    qdata):
                    skip.append(method)
                    rec_depth += 1
                    return iterate_solvers(y, n, nb, beta, nmlen, left,\
                        right, cmplx, qdata, skip, rec_depth)
                else:
                    return y
            except:
                factor = min(factor * 2, factor + 10)
        print('Method ' + method + ' unsuccessful.\n')

    raise ValueError('All methods unsuccessful.')

def verify_root(y, n, nb, beta, nmlen, left, right, cmplx, qdata) -> bool:
    '''Returns true if the root is accurate enough. False otherwise.
    '''
    zero = stpfun(y, n, nb, beta, nmlen, left, right, cmplx, qdata)

    zero = zero / la.norm(zero)
    # print('expected_zeros', np.abs(zero))
    # print('expected_zeros_sum', np.sum(np.abs(zero)))

    if np.sum(np.abs(zero)) > 10 ** (-5):
        return False
    else:
        return True

def stderiv(zp: np.array, z: np.array, beta: np.array, c: float = 1, \
    j: float = -1) -> np.array:
    '''Returns derivative at points on stripmap.'''

    fprime = np.zeros(np.shape(zp))
    npts = len(zp)
    n = len(z)

    if n != len(beta):
        raise Exception('Vector of prevertices does not contain inf values')

    ends = np.nonzero(np.isinf(z).astype(int))[0]

    theta = np.diff(beta[ends])

    if z[ends[0]] < 0:
        theta = -theta
    z = np.delete(z, ends)
    n = len(z)
    beta = np.delete(beta, ends)

    j = j - int(j > ends[0]) - int(j > ends[1])
    
    zprow = zp[np.newaxis]
    zpnew = copy.copy(zprow)
    for i in range(n - 1):
        zpnew = np.vstack([zpnew, zprow])
    
    zcol = np.transpose(z[np.newaxis])
    bcol = np.transpose(beta[np.newaxis])
    znew = copy.copy(zcol)
    bnew = copy.copy(bcol)
    for i in range(npts - 1):
        znew = np.hstack([znew, zcol])
        bnew = np.hstack([bnew, bcol])
    
    terms = np.multiply((zpnew - znew), (-np.pi / 2))

    lower = np.zeros(n)
    for i in range(n):
        if np.imag(z[i]) == 0:
            lower[i] = 1
        else:
            lower[i] = 0
    
    for i in range(n):
        if lower[i] == 1:
            terms[i] = -terms[i]
    
    rt = np.real(terms)
    terms[np.abs(rt) <= 40] = np.log(-1j * np.sinh(terms[np.abs(rt) <= 40]))
    terms[np.abs(rt) > 40] = np.multiply(np.sign(rt[np.abs(rt) > 40]), \
        terms[np.abs(rt) > 40] - 1j * np.pi / 2) - np.log(2)

    if j > 0:
        terms[(j-1), :] = terms[(j-1), :] - np.log(np.abs( \
            np.subtract(zprow, z[(j-1)])))

    return c * np.exp(np.pi / 2 * theta * zprow + np.sum(\
        np.multiply(terms, bnew), axis=0))

def stquad(z1: np.array, z2: np.array, sing1: np.array, z: np.array, \
    beta: np.array, qdata: np.array) -> np.array:
    '''Returns integral result.'''

    n = len(z)
    if len(sing1) == 0:
        sing1 = np.zeros(len(z1), dtype='complex_')

    I = np.zeros(len(z1), dtype='complex_')

    nontriv = np.nonzero((z1 != z2).astype(int))[0]

    for k in nontriv:
        za = z1[k]
        zb = z2[k]
        sng = sing1[k]

        if sng != 0:
            sng += 1

        sng = int(sng)

        dist = np.min([1, 2 * np.min(np.abs(np.hstack([z[0:sng-1], \
            z[sng:n]]) - za)) / abs(zb - za)])

        zr = za + dist * (zb - za)
        ind = np.remainder(sng + n, n + 1)
        nd = np.divide((np.multiply((zr - za), qdata[:, ind]) + zr + za), 2)

        wt = np.multiply((zr - za) / 2, qdata[:, ind+n+1])

        if 0 in np.diff(np.hstack([np.array([za]), nd, np.array([zr])])):
            I[k] = 0
        else:
            if sng > 0:
                wt = wt * (np.abs(zr - za) / 2) ** beta[sng-1]
            I[k] = np.dot(stderiv(nd, z, beta, 1, sng), wt)
            while dist < 1:
                zl = zr
                dist = np.min([1, 2 * np.min(np.abs(z - zl)) / \
                    np.abs(zl - zb)])
                zr = zl + dist * (zb - zl)
                nd = np.divide(np.multiply((zr - zl), qdata[:,n]) + zr + \
                    zl, 2)
                wt = (zr - zl) / 2 * qdata[:, 2 * n + 1]
                I[k] = I[k] + np.dot(stderiv(nd, z, beta, 1), wt)

    return I

def stquadh(z1: np.array, z2: np.array, sing1: np.array, z: np.array,\
    beta: np.array, qdata: np.array) -> np.array:
    '''Returns integral result.'''

    n = len(z)
    if len(sing1) == 0:
        sing1 = np.zeros(len(z1))

    I = np.zeros(len(z1), dtype='complex_')
    nontriv = np.nonzero(z1 != z2)[0]

    for k in nontriv:
        za = z1[k]
        zb = z2[k]
        sng = sing1[k]

        if sng != 0:
            sng += 1

        alpha = 0.75
        d = np.real(zb) - np.real(za)

        dx = np.multiply(np.real(z) - np.real(za), np.sign(d))
        dy = np.abs(np.imag(z) - np.imag(za))

        z_isinf_tfarr =  np.isinf(z)
        toright = np.logical_and(dx > 0, np.logical_not(z_isinf_tfarr))
        active = np.logical_and(dx > np.divide(dy, alpha), toright)

        sng = int(sng)
        if sng != 0:
            active[sng-1] = False
        
        x = dx[active]
        y = dy[active]

        L = np.divide((x - np.sqrt(np.square(np.multiply(alpha, x)) - \
                np.square(np.multiply(1 - alpha ** 2, np.square(y))))), \
                (1 - alpha ** 2))

        
        dy_tfarr = np.divide(np.logical_and(toright, \
            np.logical_not(active)).astype(int), alpha)
        L_test = np.hstack([L, np.nonzero(dy_tfarr)[0]])
        L = np.min(L_test)


        if L < np.abs(d):
            zmid = za + L * np.sign(d)
            I[k] = stquad(np.array([za]), np.array([zmid]), \
                np.array([sng])-1, z, beta, qdata)
            I[k] = I[k] + stquadh(np.array([zmid]), np.array([zb]), \
                np.array([0]), z, beta, qdata)
        else:
            I[k] = stquad(np.array([za]), np.array([zb]), \
                np.array([sng])-1, z, beta, qdata)
        
    return I

def scqdata(beta: np.array, nqpdts: int) -> np.array:
    '''Returns matrix of Gauss-Jacobi quadrature data.'''

    n = len(beta)
    qnode = np.zeros([nqpdts, n + 1], dtype='complex_')
    qwght = np.zeros([nqpdts, n + 1], dtype='complex_')

    for i in np.nonzero(beta > -1)[0]:
        qnode[:, i], qwght[:, i] = gaussj(nqpdts, 0, beta[i])
        
    qnode[:, n], qwght[:, n] = gaussj(nqpdts, 0, 0)

    return np.hstack((qnode, qwght))

def gaussj(n: int, alpha: float, beta: float) -> tuple:
    '''Returns nodes and weights for Gauss-Jacobi integration.'''

    apb = alpha + beta
    a = np.zeros(n)
    b = np.zeros(n)
    a[0] = (beta - alpha) / (apb + 2)
    b[0] = np.sqrt(4 * (1 + alpha) * (1 + beta) / ((apb + 3) * \
        (apb + 2) ** 2))
    
    if n > 1:
        N = np.array(range(2, n + 1))
        a[(N - 1)] = np.divide(apb * (beta - alpha), np.multiply(apb + \
            np.multiply(2, N), apb + np.multiply(2, N) - 2))
    if n > 2:
        N = np.array(range(2, n))
        b[(N - 1)] = np.sqrt(np.divide(4 * np.multiply(np.multiply(\
            np.multiply(N, N + alpha), N + beta), N + apb), np.multiply(\
            np.square(apb + 2 * N) - 1, np.square(apb + 2 * N))))
    
    if n > 1:
        D, V = la.eig(np.diag(a) + np.delete(np.delete(np.diag(b, 1), n, 0)\
            , n, 1) + np.delete(np.delete(np.diag(b, -1), n, 0), n, 1))
    else:
        V = np.ones(a.shape)
        D = a

    c = 2 ** (apb + 1) * gamma(alpha + 1) * gamma(beta + 1) / gamma(apb + 2)
    z = D
    w = c * np.square(np.transpose(V[0]))
    ind = np.argsort(z)
    z = np.sort(z)

    if n > 1:
        w = w[ind]
    else:
        w = np.array([w])

    return z, w

def stmap(zp: np.array, map) -> np.array:
    '''Helper function for the forward map.'''

    tol = map.tol
    nqpts = int(np.max([np.ceil(-np.log10(tol)), 2]))
    qdata = scqdata(map.get_beta(), nqpts)
    tol = 10 ** (-np.size(qdata, 0))
    lenzp = len(zp)
    wp = np.zeros([lenzp, 1], dtype='complex_')
    z = map.get_z()
    beta = map.get_beta()
    c = map.c

    n = len(z)

    zprow = zp[np.newaxis]
    zpnew = copy.copy(zprow)
    znew = copy.copy(np.transpose(z[np.newaxis]))
    w = map.get_w()

    for i in range(n - 1):
        zpnew = np.vstack([zpnew, zprow])
    for i in range(lenzp - 1):
        znew = np.hstack([znew, np.transpose(z[np.newaxis])])

    temp = np.abs(zpnew - znew)

    dist = np.amin(temp, axis=0)
    sing = np.argmin(temp, axis=0)

    vertex = dist < tol
    wp[vertex] = wp[sing[vertex]]
    leftend = np.logical_and(np.isinf(zp), (np.real(zp) < 0))
    wp[leftend] = w[z == -np.inf]
    rightend = np.logical_and(np.isinf(zp), (np.real(zp) > 0))
    wp[rightend] = w[z == np.inf]
    vertex = np.logical_or(vertex, np.logical_or(leftend, rightend))
    wp = np.reshape(wp, lenzp)

    atinf = np.argwhere(np.isinf(w))
    bad = np.logical_and(np.isin(sing, atinf), np.logical_not(vertex))
    
    if np.any(bad):
        print("Warning: badness in stmap.\nThis has yet to be debugged." + \
            " Take care in debugging the variable 'bad'.")
        zf = copy.copy(np.transpose(z[np.newaxis]))
        zf[np.isinf(w)] = np.inf
        zfnew = copy.copy(zf)
        for i in range(lenzp - 1):
            zfnew = np.hstack([zfnew, zf])
        

        temp = np.abs(zpnew[np.ones([n, 1]),bad] - zfnew)
        
        tmp = np.amin(temp, axis=0)
        s = np.argmin(temp, axis=0)
    
        mid1 = np.real(z[s]) + 1j / 2
        mid2 = np.real(zp[bad]) + 1j / 2
    else:
        bad = np.zeros(lenzp).astype(bool)

    zs = z[sing]
    ws = w[sing]
    normal = np.logical_and(np.logical_not(bad), np.logical_not(vertex))
    normal = np.transpose(normal)

    if np.any(normal):
        I = stquad(zs[normal],zp[normal],sing[normal], z, beta, \
            qdata)
        wp[normal] = ws[normal] + c * I
    
    if np.any(bad):
        I1 = stquad(zs[bad], mid1, sing(bad), z, beta, qdata)
        I2 = stquadh(mid1, mid2, np.zeros([sum(bad),1]), z, beta,\
            qdata)
        I3 = -stquad(zp[bad], mid2, np.zeros([sum(bad),1]), z, \
            beta, qdata)
        wp[bad] = ws[bad] + c * (I1 + I2 + I3)
    
    return wp

def stinvmap(wp: np.array, map) -> np.array:
    '''Helper function for the inverse map.'''

    tol = map.tol
    w = map.get_w()
    z = map.get_z()
    c = map.c
    n = len(w)
    lenwp = len(wp)
    zp = np.zeros(lenwp, dtype='complex_')
    beta = map.get_beta()

    nqpts = int(np.max([np.ceil(-np.log10(tol)), 2]))
    qdata = scqdata(beta, nqpts)

    done = np.zeros(lenwp, dtype='bool')

    eps = 2.2204 * 10 ** (-16)

    for i in range(n):
        idx = np.nonzero((np.abs(wp - w[i]) < 3 * eps))[0]
        zp[idx] = z[i]
        done[idx] = True
    lenwp = lenwp - np.sum(done)

    if lenwp == 0:
        return zp
    
    z0, w0 = findz0(wp, map, qdata)

    scale = wp[np.logical_not(done)] - w0

    z0 = np.hstack((np.real(z0), np.imag(z0)))


    def stimapfun(wp_stimapfun: np.array, yp_stimapfun: np.array) \
        -> np.array:
        '''Used by stinvmap for solution of ODE'''
        lenyp = len(yp_stimapfun)
        lenzp = int(lenyp / 2)

        zp = yp_stimapfun[0:lenzp] + 1j * yp_stimapfun[lenzp:lenyp]

        f = np.divide(scale, stderiv(zp, z, beta, c)[0])

        return np.concatenate((np.real(f), np.imag(f)))

    odesolver = solve_ivp(stimapfun, t_span=(0, 1), y0=z0, method='RK23')

    y = np.transpose(odesolver.y)
    t = odesolver.t

    m = np.shape(y)[0]
    leny = np.shape(y)[1]

    zp[np.logical_not(done)] = y[m-1, 0:lenwp] + 1j * y[m-1, lenwp:leny]


    zn = copy.copy(zp)
    k = 0

    maxiter = 100

    while np.sum(np.logical_not(done)) != 0 and k < maxiter:
        F = wp[np.logical_not(done)] - stmap(zn[np.logical_not(done)], map)
        dF = c * stderiv(zn[np.logical_not(done)], z, beta)[0]
        zn[np.logical_not(done)] = zn[np.logical_not(done)] + \
            np.divide(F, dF)
        done[np.logical_not(done)] = np.abs(F) < tol
        k = k + 1
    
    if np.sum(abs(F) > tol) > 0:
        raise Warning('Check solution. Warning in stinvmap.')
    
    return zn

def findz0(wp: np.array, map, qdata: np.array) -> tuple:
    '''Returns starting points for computing inverses'''
    eps = 2.2204 * 10 ** (-16)
    z = map.get_z()
    w = map.get_w()
    beta = map.get_beta()
    c = map.c

    n = len(w)
    z0 = copy.copy(wp)
    w0 = copy.copy(wp)

    atinf = np.nonzero(np.isinf(z).astype(int))[0]
    renum = np.hstack((np.arange(atinf[0], n), \
        np.arange(0, atinf[0] - 1)))
    w = w[renum]
    z = z[renum]
    beta = beta[renum]
    qdata[:, 0:n] = qdata[:, renum]
    qdata[:, (n+1):(2*n+1)] = qdata[:, (renum + n + 1)]
    kinf = np.max(np.nonzero(np.isinf(z).astype(int))[0])
    argw = np.cumsum(np.hstack((np.angle(w[2] - w[1]), -np.pi * \
        np.hstack((beta[2:n], beta[0])))))
    argw = np.hstack((argw[n-1], argw[0:(n-1)]))

    infty = np.isinf(w)
    fwd = np.roll(np.arange(0, n), -1)

    anchor = np.zeros(n, dtype='complex_')
    anchor[np.logical_not(infty)] = w[np.logical_not(infty)]
    anchor[infty] = w[fwd[infty]]

    direcn = np.exp(1j * argw)
    direcn[infty] = -direcn[infty]
    ln = np.abs(w[fwd] - w)

    factor = 0.5
    m = len(wp)
    done = np.zeros(m)
    iter = 0
    tol = 1000 * 10 ** (-np.shape(qdata)[0])

    zbase = np.empty(n, dtype='complex_')
    zbase[:] = np.nan
    wbase = copy.copy(zbase)
    idx = []

    not_finished = True

    while m > 0 and not_finished:
        for i in range(n):
            if i == 0:
                zbase[i] = np.min((-1, np.real(z[1]))) / factor
            elif i == kinf - 1:
                zbase[i] = np.max((1, np.real(z[kinf-1]))) / factor
            elif i == kinf:
                zbase[i] = 1j + np.max((1, np.real(z[kinf+1]))) / factor
            elif i == n-1:
                zbase[i] = 1j + np.min((-1, np.real(z[n-1]))) / factor
            else:
                zbase[i] = z[i] + factor * (z[i+1] - z[i])
        
            # print(stmap(np.array([zbase[i]]), map))
            wbase[i] = stmap(np.array([zbase[i]]), map)[0]

            proj = np.real((wbase[i] - anchor[i]) * np.conj(direcn[i]))
            wbase[i] = anchor[i] + proj * direcn[i]
        
        if len(idx) == 0:
            temp_wp = wp[np.logical_not(done)]
            wp_row = temp_wp[np.newaxis]
            new_wp = copy.copy(wp_row)
            
            for j in range(n-1):
                new_wp = np.vstack((new_wp, wp_row))

            wbase_row = wbase[np.newaxis]
            wbase_col = np.transpose(wbase_row)
            new_wbase = copy.copy(wbase_col)
            for j in range(m-1):
                new_wbase = np.hstack((new_wbase, wbase_col))
                
            idx = np.argmin(np.abs(new_wp - new_wbase), axis=0)

            dist = np.min(np.abs(new_wp - new_wbase), axis=0)
        
        else:
            not_done = np.logical_not(done)

            idx[not_done] = np.remainder(idx[not_done], n) + 1
            for i in range(len(idx)):
                if idx[i] >= len(wbase):
                    idx[i] = 0
            

        
        # print(idx)

        not_done = np.logical_not(done)
        print(not_done)
        w0[not_done] = wbase[idx[not_done]]
        z0[not_done] = zbase[idx[not_done]]

        for i in range(n):
            # print(idx == i)
            active = np.logical_and(idx == i, not_done)

            if np.any(active):
                done[active] = np.ones(np.sum(active))

                for k in range(n):
                    if k == i:
                        continue
                    A = np.array([np.real(direcn[k]), np.imag(direcn[k])])
                    findactive = np.nonzero(active)[0]
                    # print(findactive)
                    for p in findactive:
                        dif = w0[p] - wp[p]
                        # print(dif)
                        temp_A_add = np.array([np.real(dif), np.imag(dif)])
                        temp_A_add = np.reshape(temp_A_add, np.shape(A))

                        # print('A', A)
                        # print('B', temp_A_add)

                        A_new = np.vstack((A, temp_A_add))
                        A_new = np.transpose(A_new)
                        # print(A_new)
                        rcondA = 1 / np.linalg.cond(A_new)

                        if rcondA < eps:
                            wpx = np.real((wp[p] - anchor[k]) / direcn[k])
                            w0x = np.real((w0[p] - anchor[k]) / direcn[k])

                            if (wpx * w0x < 0) or \
                                ((wpx - ln[k]) * (w0x - ln[k]) < 0):
                                # print('line 980')
                                done[p] = 0
                            
                        else:
                            dif = w0[p] - anchor[k]
                            b = np.array([[np.real(dif)], [np.imag(dif)]])
                            s = la.solve(A_new, b)

                            if s[0] >= 0 and s[0] <= ln[k]:
                                if np.abs(s[1] - 1) < tol:
                                    z0[p] = zbase[k]
                                    w0[p] = wbase[k]
                                elif np.abs(s[1]) < tol: 
                                    if np.real(np.conj(\
                                        wp[p] - w0[p]) * 1j * direcn[k]) > 0:
                                        done[p] = 0
                                        # print('line 995')
                                elif s[1] > 0 and s[1] < 1:
                                    # print('line 998')
                                    done[p] = 0
            
                m = sum(np.logical_not(done))
                if m == 0:
                    not_finished = False
                    break
        
        if iter > 2 * n:
            raise Exception('Error has occurred.')
        else:
            iter = iter + 1
        factor = random.random()
    

    return z0, w0



