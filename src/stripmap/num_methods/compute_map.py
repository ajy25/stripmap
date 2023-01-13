import copy
import numpy as np
from scipy.integrate import RK23
from num_methods.solve_param import scqdata, stquad, stquadh, stderiv

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
    n = len(w)
    lenwp = len(wp)
    zp = np.zeros(lenwp, dtype='complex_')
    beta = map.get_beta()

    nqpts = int(np.max([np.ceil(-np.log10(tol)), 2]))
    qdata = scqdata(beta, nqpts)

    done = np.zeros(lenwp, dtype='bool')

    eps = 2.2204 * 10 ** (-16)

    for i in range(n):
        idx = np.nonzero((np.abs(wp - w[i]) < 3 * eps).as_type(int))[0]
        zp[idx] = z[i]
        done[idx] = True
    lenwp = lenwp - np.sum(done.as_type(int))

    
    return None

def findz0(wp: np.array, map, qdata: np.array) -> tuple:
    '''Returns starting points for computing inverses'''
    z = map.get_z()
    w = map.get_w()
    beta = map.get_beta()
    c = map.c

    n = len(w)
    z0 = wp
    w0 = wp

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
        np.hstack(beta[2:n], beta[0]))))
    argw = np.hstack((argw[n-1], argw[1:n-1]))

    infty = np.isinf(w)
    fwd = np.roll(np.arange(1, n+1), -1)

    anchor = np.zeros(n, dtype='complex_')
    anchor[np.logical_not(infty)] = w[np.logical_not(infty)]
    anchor[infty] = w[fwd[infty]]

    direcn = np.exp(1j * argw)
    direcn[infty] = -direcn[infty]
    ln = np.abs(w[fwd] - w)

    factor = 0.5
    m = len(wp)
    done = np.zeros(1, m)
    iter = 0
    tol = 1000 * 10 ** (-np.shape(qdata)[0])

    zbase = np.empty(n)
    zbase[:] = np.nan
    wbase = copy.copy(zbase)
    idx = []

    while m > 0:
        for i in range(n):
            if i == 0:
                zbase[i] = np.min((-1, np.real(z[1]))) / factor
            elif i == kinf - 1:
                zbase[i] = np.max((1, np.real(z[kinf-1]))) / factor
            elif i == kinf:
                zbase[i] = 1j + np.max((1, np.real(z[kinf+1]))) / factor
            elif i == n:
                zbase[i] = 1j + np.min((-1, np.real(z[n-1]))) / factor
            else:
                zbase[i] = z[i] + factor * (z[i+1] - z[i])
        
            wbase[i] = stmap(zbase[i], map)

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

    return z0, w0

def stimapfun(yp: np.array, z: np.array, scale, beta: np.array, c: float) \
    -> np.array:
    '''Used by stinvmap for solution of ODE'''
    lenyp = len(yp)
    lenzp = np.floor(lenyp / 2)

    zp = yp[:lenzp] + 1j * yp[lenzp:]
    f = np.divide(scale, stderiv(zp, z, beta, c))
    return np.concatenate((np.real(f), np.imag(f)))

