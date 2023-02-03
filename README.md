# viz_project

Reimplementing [ShapeAsPoints](https://github.com/autonomousvision/shape_as_points) with documentation and examples.

Currently requires (list may be incomplete):
- `pytorch3d`: mostly for the loss function (pytorch3d.loss.chamfer_distance)
- `scikit-image`: marching cubes, may later switch to pytorch3d version if normal estimation is decent (and C++ version is fixed)
- `torch`: backpropagation
- `meshplot`: simple interactive plots
- `pandas`: load datasets
