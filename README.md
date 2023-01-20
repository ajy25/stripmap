# stripmap 

This repository is an attempt at translating the stripmap class of the [Schwarz-Christoffel Toolbox for conformal mapping](https://github.com/tobydriscoll/sc-toolbox) from MATLAB to Python. This toolbox was developed by Tobin A Driscoll; its user guide is linked [here](https://tobydriscoll.net/project/sc-toolbox/guide.pdf). Details regarding numerical methods for solving the side-length parameter problem are found in *Schwarz-Christoffel Mapping* by Driscoll and Trefethen. 

## Usage
Import the Stripmap and Polygon classes.
```
from map import Stripmap, Polygon
```


Initialize a Polygon with counterclockwise vertices.
```
x = [1, 2, 1, 0, 0, -1]  # example x vertices
y = [-1, 0, 1, 1, 0, -1] # example y vertices
p = Polygon(x, y)
```

Initialize a Stripmap; prevertices are automatically computed. Prevertex computation needs to be more thoroughly tested.
```
ends = [1, 4] # example of ends (one-indexed)
m = Stripmap(p, ends)
```

Compute inverse map
```
interior_points = ...
mapped_points = map.evalinv(interior_points)
```

## Requirements

Python 3.8+, NumPy, SciPy

## Testing

Testing was done directly against Driscoll's MATLAB Schwarz-Christoffel Toolbox for conformal mapping. 
Random polygons were generated, and the outputs from this Python package and the MATLAB toolbox were compared.
See the testing folder for more details. 

## References 

Driscoll, T., & Trefethen, L. (2002). *Schwarz-Christoffel Mapping* (Cambridge Monographs on Applied and Computational Mathematics). Cambridge: Cambridge University Press. doi:10.1017/CBO9780511546808
