import numpy as np
from stripmap import Stripmap, Polygon


x = [1, 2, 1, 0, 0, -1]
y = [-1, 0, 1, 1, 0, -1]
poly = Polygon(x, y)

ends = [1, 5]

map = Stripmap(poly, ends)
print(map.eval([1, 1], [0, 1]))

n = 5
print(np.roll(np.arange(1, n+1), -1))