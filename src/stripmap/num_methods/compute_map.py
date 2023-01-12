import copy
import numpy as np
from num_methods.solve_param import *

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

    tol = map_tol
    lenwp = len(wp)
    zp = np.zeros(lenwp, dtype='complex_')
    beta = map.get_beta()

    nqpts = int(np.max([np.ceil(-np.log10(tol)), 2]))
    qdata = scqdata(beta, nqpts)

    return None
