import numpy as np
from polygon import Polygon
from map import Stripmap


x = [1, 2, 1, 0, 0, -1]
y = [-1, 0, 1, 1, 0, -1]
poly = Polygon(x, y)

ends = [1, 4]

map = Stripmap(poly, ends)