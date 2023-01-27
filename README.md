# stripmap 

This repository is a rewrite of the stripmap class of the MATLAB [Schwarz-Christoffel Toolbox for conformal mapping](https://github.com/tobydriscoll/sc-toolbox) in Python. This toolbox was developed by Tobin A Driscoll; its user guide is linked [here](https://tobydriscoll.net/project/sc-toolbox/guide.pdf). Details regarding numerical methods for solving the side-length parameter problem are found in *Schwarz-Christoffel Mapping* by Driscoll and Trefethen. 

## Installation
```
pip install shapely
pip install -i https://test.pypi.org/simple/ stripmap
```

## Usage
Import the Stripmap and Polygon classes.
```
from stripmap.map import Stripmap, Polygon
```

Initialize a Polygon with counterclockwise vertices.
```
x = [0, 0.5, 1, 1.5, 2, 0, -1, -1.5, -2, -2]    # example x vertices
y = [2, 4, 6, 10, 12, 10, 8, 4, 1, 0]           # example y vertices
test_poly = Polygon(x, y)
```

Initialize a Stripmap. Prevertices are automatically computed.
```
ends = [1, 6]                                   # example of ends (one-indexed)
test_map = Stripmap(test_poly, ends)
print(test_map)                                 # print prevertices and constant
```

Compute the forward map.
```
x_in_poly = [0.1, 0.2, -6.5]
y_in_poly = [0.9, 0.8, 0]
x_mapped, y_mapped = test_map.eval(x_in_poly, y_in_poly)
print(x_mapped)
print(y_mapped)
```

Compute the inverse map.
```
x_in_poly = [0, -1, -1.3]
y_in_poly = [6, 2, 2.1]
x_mapped, y_mapped = test_map.evalinv(x_in_poly, y_in_poly)
print(x_mapped)
print(y_mapped)
```

## Requirements

Python 3.8+, NumPy, SciPy

## Note regarding testing

Testing was done directly against Driscoll's MATLAB Schwarz-Christoffel Toolbox for conformal mapping. 
Random polygons were generated, and the outputs from this Python package and the MATLAB toolbox were compared.

## References 

Driscoll, T., & Trefethen, L. (2002). *Schwarz-Christoffel Mapping* (Cambridge Monographs on Applied and Computational Mathematics). Cambridge: Cambridge University Press. doi:10.1017/CBO9780511546808