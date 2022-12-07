# %%

import numpy as np
from polygon import Polygon
from stripmap import Stripmap


x = np.array([1, 2, 1, 0, 0, -1])
y = np.array([-1, 0, 1, 1, 0, -1])


test_poly =  Polygon(x, y)

alpha = test_poly.get_angles()
w = test_poly.get_vertices()
beta = alpha - 1
ends = np.array([1, 4])



# 


# %%

test_map = Stripmap(test_poly, [1, 4])


# print(y0, n, nb, beta, nmlen, left, right, cmplx, qdata)

# test_map.stpfun(y0, n, nb, beta, nmlen, left, right, cmplx, qdata)

# print(test_map.stquadh(z1, z2, sing1, z, beta, qdata))
# %%
