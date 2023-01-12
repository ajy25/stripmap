if __name__ == '__main__':
    x_vert = np.array([0, 0.5, 1, 1.5, 2, 0, -1, -1.5, -2, -2])
    y_vert = np.array([2, 4, 6, 10, 12, 10, 8, 4, 1, 0])

    test_poly =  Polygon(x_vert, y_vert)

    test_map = Stripmap(test_poly, [1, 6])

    y1 = np.array([-0.031214892317576, 0.532128982172614, -2.758616578522062, 1.671321856365236, 0.605751389975247, -0.221680763461464, -5.516275062642263])
    n = 8
    nb = 4
    beta = test_map.beta
    nmlen = np.array([-0.41573034-0.13483146j, 0.85459761+0.j, 0.43704832+0.j, \
        0.85459761+0.j, 0.64477154+0.j, 0.21199958+0.j])
    left = np.array([0, 0, 1, 2, 4, 5, 6])
    right = np.array([7, 1, 2, 3, 5, 6, 7])
    cmplx = np.array([False, True, False, False, False, False, False])
    qdata = test_map.qdata


    left = left + 1
    right = right + 1
    

    # print('y0: ', y1)
    # print('n: ', n)
    # print('nb: ', nb)
    # print('beta: ', beta)
    # print('nmlen: ', nmlen)
    # print('left: ', left)
    # print('right: ', right)
    # print('cmplx: ', cmplx)
    print('qdat: ', np.sum(qdata))
    print('qdatshape: ', np.shape(qdata))


    eng = matlab.engine.start_matlab()
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

    # print(test_map.stpfun(y1, n, nb, beta, nmlen, left, right, cmplx, qdata))