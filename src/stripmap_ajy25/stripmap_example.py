from map import Polygon, Stripmap
import matplotlib.pyplot as plt

x_vert = [0, 0.5, 1, 1.5, 2, 0, -1, -1.5, -2, -2]
y_vert = [2, 4, 6, 10, 12, 10, 8, 4, 1, 0]

test_poly =  Polygon(x_vert, y_vert)
test_map = Stripmap(test_poly, [0, 5])

test_map.plot_poly()

print(test_map.eval([0.1, 0.2, -6.5], [0.9, 0.8, 0]))
print(test_map.evalinv([0, -1, -1.3], [6, 2, 2.1]))

