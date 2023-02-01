# viz_project

Reimplementing [ShapeAsPoints](https://github.com/autonomousvision/shape_as_points) with cleaner code and actual documentation.

Currently requires (list may be incomplete):
- pytorch3d: mostly for the loss function (pytorch3d.loss.chamfer_distance)
- scikit-image: for marching cubes, may later switch to pytorch3d if normal estimation is decent
- torch: backpropagation
- meshplot: simple interactive plots
- pandas: load datasets
