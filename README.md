# stripmap 

This repository is an attempt at translating the stripmap class of the [Schwarz-Christoffel Toolbox for conformal mapping](https://github.com/tobydriscoll/sc-toolbox) from MATLAB to Python. This toolbox was developed by Tobin A Driscoll; its user guide is linked [here](https://tobydriscoll.net/project/sc-toolbox/guide.pdf). Details regarding numerical methods for solving the side-length parameter problem are found in *Schwarz-Christoffel Mapping* by Driscoll and Trefethen. 

## Usage

Initialize a Polygon with counterclockwise vertices.
```
x = np.array([1, 2, 1, 0, 0, -1])
y = np.array([-1, 0, 1, 1, 0, -1])
poly = Polygon(x, y)
```

Initialize a Stripmap; prevertices are automatically computed. Prevertex computation needs to be more thoroughly tested.
```
map = Stripmap(poly)
```

Compute inverse map *(to be implemented in the near future)*
```
interior_points = ...
mapped_points = map.evalinv(interior_points)
```

## Requirements

Python 3.8+, NumPy, SciPy

## References 

Driscoll, T., & Trefethen, L. (2002). *Schwarz-Christoffel Mapping* (Cambridge Monographs on Applied and Computational Mathematics). Cambridge: Cambridge University Press. doi:10.1017/CBO9780511546808
