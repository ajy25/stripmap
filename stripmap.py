import numpy as np
import copy
import scipy.linalg as la
from scipy.special import gamma
from scipy.optimize import fsolve
from polygon import Polygon

class Stripmap:
    # fields
    p = None
    z = None
    c = None
    beta = None
    ends = None
    qdata = None
    tol = 10 ** (-8)

    # methods
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
        self.stparam()
        
        # method end
        return

    def stparam(self):
        '''Attempts to solve for the parameters of the stripmap'''

        # define necessary local variables
        N = len(self.beta)                      # number of vertices
        w = copy.copy(self.p.get_vertices())    # vertices
        beta = copy.copy(self.beta)             # betas
        n = N - 2                               # number of prevertices

        renum = np.hstack([np.arange(self.ends[0], N), \
            np.arange(self.ends[0])])

        k = np.nonzero(renum == self.ends[1])[0][0] + 1
        
        # number of prevertices on the bottom edge of the strip
        nb = k - 2

        # renumber vertices to start with first element of ends
        w = w[renum]
        beta = beta[renum]

        # quadrature data
        nqpts = int(np.max([np.ceil(-np.log10(self.tol)), 4]))
        qdata = self.scqdata(self.beta, nqpts)
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

        y = fsolve(self.stpfun, np.real(y0), (n, nb, beta, nmlen, left, right, cmplx, \
            qdata))
        print(y)
        
        z = np.zeros(n, dtype='complex_')
        z[1:nb] = np.cumsum(np.exp(y[0:(nb-1)]))
        z[nb:] = np.add(np.cumsum(np.hstack([np.array([y[nb-1]]), \
            -np.exp(y[nb:n])])), 1j)

        z = np.hstack([np.array([-np.infty]), z[0:nb], np.array([np.infty]), \
            z[nb:n]])
        
        mid = np.mean(z[1:3])
    
        print(z)

        g = self.stquad(np.array([z[2]]), np.array([mid]), np.array([2]), z, beta, qdata) - \
            self.stquad(np.array([z[1]]), np.array([mid]), np.array([1]), z, beta, qdata)

        c = np.divide(w[0] - w[1], g)

        z[renum] = z
        qdata[:, renum] = qdata[:, np.arange(N)]
        qdata[:, N + 1 + renum] = qdata[:, N + 1 + np.arange(N)]

        # set up determined params

        print(c)

        self.z = z
        self.c = c
        self.qdata = qdata

        # method end
        return
    
    def stpfun(self, y: np.array, n: int, nb: int, beta: np.array, nmlen: int, \
        left: np.array, right: np.array, cmplx: np.array, qdata: np.array):
            z = np.zeros(n, dtype='complex_')
            z[1:nb] = np.cumsum(np.exp(y[0:(nb-1)]))
            z[nb:] = 1j + np.cumsum(np.vstack([[y[nb-1]], -np.exp(y[nb:(n-1)])]))

            zleft = z[left]
            zright = z[right]
            mid = np.mean(np.vstack([zleft, zright]), axis=0)

            # cmplx is logical; so is c2
            c2 = copy.copy(cmplx)
            c2[1] = False
            mid[c2] = mid[c2] - np.sign(left[c2]) * 1j / 2

            zs = np.hstack([np.array([-np.infty]), z[0:nb], \
                np.array([np.infty]), z[nb:n]])
            left = np.add(left, np.add(1, (left > nb-1).astype(int)))
            right = np.add(right, np.add(1, (right > nb-1).astype(int)))
            
            ints = np.zeros(n - 1, dtype='complex_')
            c2[0] = True
            id = np.logical_not(c2)

            ints[id] = self.stquadh(zleft[id], mid[id], left[id], zs, beta, \
                qdata) - self.stquadh(zright[id], mid[id], right[id], zs, \
                beta, qdata)

            z1 = np.add(np.real(zleft[c2]), 1j / 2)
            z2 = np.add(np.real(zright[c2]), 1j / 2)
            id = np.logical_not(id)

            ints[id] = self.stquad(zleft[id], z1, left[id], zs, beta, qdata)


            ints[id] = ints[id] + self.stquadh(z1, z2, np.zeros(len(z1)), zs, \
                beta, qdata)

            ints[id] = ints[id] - self.stquad(zright[id], z2, right[id], zs, \
                beta, qdata)
            
            absval = np.abs(ints[np.logical_not(cmplx)])

            if absval[0] == 0:
                rat1 = 0
                rat2 = 0
            else:
                rat1 = np.divide(absval[1:], absval[0])
                rat2 = np.divide(ints[cmplx], ints[0])

            rat_test = np.hstack([rat1, rat2])
            
            if 0 in rat_test or np.any(np.isinf(rat_test)) or \
                np.any(np.isnan(rat_test)):
                print('WARNING: SEVERE CROWDING')
            
            cmplx2 = cmplx[1:]

            if len(rat1) != 0:
                F1 = np.log(np.divide(rat1, nmlen[np.logical_not(cmplx2)]))
            else:
                F1 = np.array([])
            
            if len(rat2) != 0:
                F2 = np.log(np.divide(rat2, nmlen[cmplx2]))
            else:
                F2 = np.array([])

            print(np.real(np.hstack([F1, np.real(F2), np.imag(F2)])))
            return np.real(np.hstack([F1, np.real(F2), np.imag(F2)]))

    def stderiv(self, zp: np.array, z: np.array, beta: np.array, c: float, \
        j: float = -1) -> np.array:
        '''Returns derivative at points on stripmap'''

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

    def stquad(self, z1: np.array, z2: np.array, sing1: np.array, z: np.array, \
        beta: np.array, qdata: np.array) -> np.array:
        '''Returns integral result'''
        n = len(z)
        if len(sing1) == 0:
            sing1 = np.zeros(len(z1), dtype='complex_')

        I = np.zeros(len(z1), dtype='complex_')

        nontriv = np.nonzero((z1 != z2).astype(int))[0]

        for k in nontriv:
            za = z1[k]
            zb = z2[k]
            sng = sing1[k] + 1
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
                I[k] = np.dot(self.stderiv(nd, z, beta, 1, sng), wt)
                while dist < 1:
                    zl = zr
                    dist = np.min([1, 2 * np.min(np.abs(z - zl)) / \
                        np.abs(zl - zb)])
                    zr = zl + dist * (zb - zl)
                    nd = np.divide(np.multiply((zr - zl), qdata[:,n]) + zr + \
                        zl, 2)
                    wt = (zr - zl) / 2 * qdata[:, 2 * n + 1]
                    I[k] = I[k] + np.dot(self.stderiv(nd, z, beta, 1), wt)

        return I

    def stquadh(self, z1: np.array, z2: np.array, sing1: np.array, z: np.array,\
        beta: np.array, qdata: np.array):
        '''Returns integral result'''
        n = len(z)
        if len(sing1) == 0:
            sing1 = np.zeros(len(z1))
        
        # print(z1, z2, sing1, z, beta, qdata)

        I = np.zeros(len(z1), dtype='complex_')
        nontriv = np.nonzero((z1 != z2).astype(int))[0]

        # print(I, nontriv)

        for k in nontriv:
            za = z1[k]
            zb = z2[k]
            sng = sing1[k] + 1

            alpha = 0.75
            d = np.real(zb) - np.real(za)

            dx = np.multiply(np.real(z) - np.real(za), np.sign(d))
            dy = np.abs(np.imag(z) - np.imag(za))

            z_isinf_tfarr =  np.isinf(z)
            dx_tfarr = dx > 0
            toright = np.logical_and(dx_tfarr, np.logical_not(z_isinf_tfarr))
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
                I[k] = self.stquad(np.array([za]), np.array([zmid]), \
                    np.array([sng])-1, z, beta, qdata)
                I[k] = I[k] + self.stquadh(np.array([zmid]), np.array([zb]), \
                    np.array([0]), z, beta, qdata)
            else:
                I[k] = self.stquad(np.array([za]), np.array([zb]), \
                    np.array([sng])-1, z, beta, qdata)
            
        return I

    def scqdata(self, beta: np.array, nqpdts: int) -> np.array:
        '''Returns matrix of Gauss-Jacobi quadrature data'''
        n = len(beta)
        qnode = np.zeros([nqpdts, n + 1], dtype='complex_')
        qwght = np.zeros([nqpdts, n + 1], dtype='complex_')

        for i in np.nonzero(beta[beta > -1])[0]:
            qnode[:, i], qwght[:, i] = self.gaussj(nqpdts, 0, beta[i])
        qnode[:, n], qwght[:, n] = self.gaussj(nqpdts, 0, 0)
        return np.hstack((qnode, qwght))

    def gaussj(self, n: int, alpha: float, beta: float) -> tuple:
        '''Returns nodes and weights for Gauss-Jacobi integration'''
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

    def stinv(self) -> None:
        '''Inverse of the map'''
        return

    def info(self) -> str:
        '''Displays information regarding the map'''
        return 