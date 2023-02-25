# viz_project

Reimplementing [ShapeAsPoints](https://github.com/autonomousvision/shape_as_points) with documentation and examples.

Currently requires (list may be incomplete):
- `pytorch3d`: mostly for the loss function (pytorch3d.loss.chamfer_distance)
- `scikit-image`: marching cubes, may later switch to pytorch3d version if normal estimation is decent (and C++ version is fixed)
- `torch`: backpropagation
- `meshplot`: simple interactive plots
- `pandas`: load datasets

Usage:

To run the optimization method on an initial point cloud it is sufficient to run the following lines:
```
from sap import ShapeAsPoints
sap = ShapeAsPoint(
  initial_points,
  initial_points_normals, # can even be random
  target_points,
  device="device_of_choice" # preferably a CUDA capable GPU
)
# for each grid resolution we can then specify how many epoch to run
sap.train(
  [(32, 1000), (64, 500), ] # 1000 epochs at 32^3 voxels, 500 at 64^3
)
```
It is also possible to add a callback plot or save records after each epoch. The callback should accept a dictionary containing:
- `epoch` current epoch
- `chi` indicator grid (a tensor of shape [1, *grid_resolution])
- `mesh` a triple of vertices, faces, vertex normals
- `loss` current epoch loss value
