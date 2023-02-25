# viz_project

Reimplementing [ShapeAsPoints](https://github.com/autonomousvision/shape_as_points) with documentation and examples.

Implementation currently requires:
- `pytorch3d`: chamfer loss (pytorch3d.loss.chamfer_distance) and initial starting sphere
- `scikit-image`: marching cubes (pytorch3d version has no normal estimation and C++ (cpu) version is bugged)
- `torch`: for autograd backpropagation and efficient gpu computation
Running Notebooks additionally requires (list may be incomplete):
- `meshplot`: interactive plots
- `pandas`: to load xyz datasets
- `ipywidgets` : interactable widgets

The easiest way to install all requirements, albeit no the most efficient, is to run:
```
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install pandas plyfile ipywidgets pythreejs "git+https://github.com/skoch9/meshplot.git"
```

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

Code Structure:
- Notebook 1 to 3 explain various key aspects of the optimization approach.
- Notebook 4 shows an example morphing a sphere to a target.
- implementation contains legacy code for notebook 1 to 3.
- sap contains the actual implementation code used in notebook 4.
