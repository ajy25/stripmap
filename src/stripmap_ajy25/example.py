import numpy as np
from map import Stripmap, Polygon


x_vert = np.array([0, 0.5, 1, 1.5, 2, 0, -1, -1.5, -2, -2])
y_vert = np.array([2, 4, 6, 10, 12, 10, 8, 4, 1, 0])

wp_x_vert = np.array([0, -1])
wp_y_vert = np.array([6, 2])
wp = np.array([6j, -1 + 2j])

test_poly =  Polygon(x_vert, y_vert)

test_map = Stripmap(test_poly, [1, 6])
qdata = test_map.get_qdata()

print(test_map.eval([0.1, 0.2, -6.5], [0.9, 0.8, 0]))
print(test_map.evalinv(wp_x_vert, wp_y_vert))
