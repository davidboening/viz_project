# python libraries
from numbers import Number
from typing import Tuple, Optional, Callable, Sequence

# external dependencies
import torch
from pytorch3d.loss import chamfer_distance
from skimage.measure import marching_cubes


def minmax_scaling(points: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Scale all points to be within [0,1]

    Parameters
    ----------
    points : torch.tensor
        shape = (batchsize, npoints, dim)
    eps : float
        small adjustement to min and max

    Returns
    -------
    torch.tensor
        all coordinates within [0,1]
    """
    min = points.min(axis=1)[0].min() - eps
    max = points.max(axis=1)[0].max() + eps
    return (points - min) / (max - min)


def unitsphere_scaling(points: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Center input in origin and shift in [-1,1] * scale.

    Parameters
    ----------
    points : torch.Tensor
        points to process
    scale : float, optional
        scale, by default 1.0

    Returns
    -------
    torch.Tensor
        scaled points
    """
    center = points.mean(axis=1)
    shifted = points - center
    scaleto1 = torch.abs(shifted).max(axis=1)[0].max()
    zoomed = shifted / (scale * scaleto1)
    return zoomed


def get_fft_frequencies(grid: torch.Tensor) -> torch.Tensor:
    """Returns FFT frequencies for a given grid.

    Parameters
    ----------
    grid : torch.Tensor
        grid dimensions eg. [32, 32, 32] for a 3-dimensional grid 
        with 32 samples along each dimension

    Returns
    -------
    torch.Tensor
        shape = [*grid, 3]
    """
    freqs = []
    for res in grid:
        freqs.append(torch.fft.fftfreq(res, d=1 / res))
    freqs = torch.stack(torch.meshgrid(freqs, indexing="ij"), dim=-1)
    return freqs


def get_gaussian_smoothing(
    input: torch.Tensor, sigma: Number = 5, res: Optional[Number] = None
) -> torch.Tensor:
    """Returns a gaussian smoothing kernel

    Parameters
    ----------
    input : torch.Tensor
        smoothing input, shape = [*grid]
    sigma : Number, optional
        bandwidth, by default 5
    res : Optional[Number], optional
        resolution, by default None (grid[0])

    Returns
    -------
    torch.Tensor
        shape = [*grid]
    """
    if res is None:
        res = input.shape[0]
    _vector = torch.sum(input**2, dim=-1)
    _scalar = -2 * (sigma / res) ** 2
    return torch.exp(_scalar * _vector)


def point_rasterization(
    points: torch.Tensor, features: torch.Tensor, grid: torch.Tensor
) -> torch.Tensor:
    """Returns a scalar/vector field as a grid of specified dimension.
    Values of the field are given by features while positions by points.
    Trilinear interpolation is used to approximate values in grid points.
    ATTENTION: points must be in [0, 1], changing center is not supported

    Parameters
    ----------
    points : torch.Tensor
        batch of sample positions
    features : torch.Tensor
        batch of sample values
    grid : torch.Tensor
        field resolution, shape = [points.shape[-1]]

    Returns
    -------
    torch.Tensor
        scalar/vector field of shape [batch, nfeatures, *grid]
    """
    assert (points.min() >= 0 and points.max() <= 1), "query points outside grid"
    device = points.device
    batchsize = points.shape[0]
    samplesize = points.shape[1]
    dim = points.shape[2]
    featuresize = features.shape[2]
    # x0, y0, z0
    # gridstart = torch.tensor([0.]*dim)
    voxelcount = grid
    
    # s0, s1, s2
    voxelsize = 1 / voxelcount
    eps = 1e-5

    # compute neighbor indices
    lower_index = torch.floor(points / voxelsize).int()
    # lower_index = lower_index.remainder(voxelcount)

    upper_index = torch.ceil(points / voxelsize).int()
    upper_index = upper_index.remainder(voxelcount)

    sample_index = torch.stack([lower_index, upper_index], dim=0)

    # all subsets of 2**dim elements
    # e.g. if dim=3 : 000, 001, 010, ..., 110, 111
    combinations = torch.stack(
        torch.meshgrid(
            *([torch.tensor([0, 1], device=device)] * dim), 
            indexing="ij"
        ), dim=-1
    ).reshape(2**dim, dim)

    # [0,1,..,dim-1] * (2**dim)
    # e.g if dim=3 : [[1,2,3],[1,2,3],...]
    selection = torch.arange(dim).repeat(2**dim, 1)

    # creates all possible indexing combinations
    # e.g. (low, low, low), (low, low, up), ..., (up, up, up)
    # combinations generates all possible combinations:
    # (xl, xl, xl), (xl, xl, xu), ...
    # (yl, yl, yl), (yl, yl, yu), ...
    # (zl, zl, zl), (zl, zl, zu), ...
    # selection then selects the diagonal of each:
    # -> (xl, yl, zl), (xl, yl, zu), (xl, yu, zl), ...
    neighbor_index = sample_index[combinations, ..., selection]
    sample_index = neighbor_index.permute(2, 3, 0, 1)  # [batch, npoints, 2**dim, dim]

    # similarly we construct the cube coordinates
    lower_coords = lower_index * voxelsize
    upper_coords = (lower_index + 1) * voxelsize
    sample_coords = torch.stack([lower_coords, upper_coords], dim=0)
    # here we invert the order from upper to lower instead
    neighbor_coords = sample_coords[1 - combinations, ..., selection]
    neighbor_coords = neighbor_coords.permute(2, 3, 0, 1)

    # distances from points to neighbors
    neighbor_dist = torch.abs(points.unsqueeze(-2) - neighbor_coords)
    # scale distances to use cubes as a metric
    neighbor_dist /= voxelsize

    # compute trilinear weights for all 8 neighbors of each sample as:
    # n0 = |x_{n0} - x|/sx * |y_{n0} - y|/sy * |z_{n0} - z|/sz * (feature_value)
    weights = torch.prod(neighbor_dist, dim=-1, keepdim=False)
    field_values = weights.unsqueeze(-1) * features.unsqueeze(-2)

    # initialize batch and feature indices
    batch_index = torch.arange(batchsize, device=device).expand(samplesize, 2**dim, batchsize)
    batch_index = batch_index.permute(2, 0, 1)
    feature_index = torch.arange(featuresize, device=device).reshape(1, 1, 1, featuresize, 1)
    # solve broadcasting issues
    batch_index = batch_index.unsqueeze(-1).unsqueeze(-1)
    batch_index = batch_index.expand(batchsize, samplesize, 2**dim, featuresize, 1)
    sample_index = sample_index.unsqueeze(-2)
    sample_index = sample_index.expand(
        batchsize, samplesize, 2**dim, featuresize, dim
    )
    feature_index = feature_index.expand(
        batchsize, samplesize, 2**dim, featuresize, 1
    )
    # construct final index
    index = torch.cat([batch_index, feature_index, sample_index], dim=-1)

    # flatten all dimensions
    index = index.reshape(-1, dim + 2)
    field_values = field_values.reshape(-1)

    # construct output grid
    output_size = torch.Size((batchsize, featuresize, *grid))
    output_grid = torch.zeros(output_size, dtype=field_values.dtype, device=device).view(-1)

    # flatten the index
    index_folds = torch.tensor([1] + list(output_size[:0:-1]), device=device).cumprod(0).flip(0)
    index_flat = torch.sum(index * index_folds, dim=-1)

    # write field values to grid at index position
    output_grid.scatter_add_(0, index_flat, field_values)
    output_grid = output_grid.view(*output_size)

    return output_grid


def grid_interpolation(
    field_values: torch.Tensor, query_points: torch.Tensor
) -> torch.Tensor:
    """Given a scalar/vector field approximated using a grid and a set
    of query points returns field values for each query point approximated
    using trilinear interpolation.
    ATTENTION: query_points must be in [0, 1]

    Parameters
    ----------
    field_values : torch.Tensor
        a batch of field grid with shape = [batch, *grid, nfeatures]
    query_points : torch.Tensor
        a batch of query points with shape = [batch, npoints, dim]

    Returns
    -------
    torch.Tensor
        field values at query points with shape = [batch, npoints, nfeatures]
    """
    assert (query_points.min() >= 0 and query_points.max() <= 1), "query points outside grid"
    device = query_points.device
    batchsize = query_points.shape[0]
    samplesize = query_points.shape[1]
    dim = query_points.shape[2]
    voxelcount = torch.tensor(field_values.shape[1:-1], device=device)
    # s0, s1, s2
    voxelsize = 1 / voxelcount
    eps = 1e-5

    # compute neighbor indices
    lower_index = torch.floor(query_points / voxelsize).int()
    # lower_index = lower_index.remainder(voxelcount)

    upper_index = torch.ceil(query_points / voxelsize).int()
    upper_index = upper_index.remainder(voxelcount)

    sample_index = torch.stack([lower_index, upper_index], dim=0)

    # all subsets of 2**dim elements
    # e.g. if dim=3 : 000, 001, 010, ..., 110, 111
    combinations = torch.stack(
        torch.meshgrid(
            *([torch.tensor([0, 1], device=device)] * dim), 
            indexing="ij"
        ), dim=-1
    ).reshape(2**dim, dim)

    # [0,1,..,dim-1] * (2**dim)
    # e.g if dim=3 : [[1,2,3],[1,2,3],...]
    selection = torch.arange(dim, device=device).repeat(2**dim, 1)

    # creates all possible indexing combinations
    # e.g. (low, low, low), (low, low, up), ..., (up, up, up)
    # combinations generates all possible combinations:
    # (xl, xl, xl), (xl, xl, xu), ...
    # (yl, yl, yl), (yl, yl, yu), ...
    # (zl, zl, zl), (zl, zl, zu), ...
    # selection then selects the diagonal of each:
    # -> (xl, yl, zl), (xl, yl, zu), (xl, yu, zl), ...
    neighbor_index = sample_index[combinations, ..., selection]
    sample_index = neighbor_index.permute(2, 3, 0, 1)  # [batch, npoints, 2**dim, dim]

    # similarly we construct the cube coordinates
    lower_coords = lower_index * voxelsize
    upper_coords = (lower_index + 1) * voxelsize
    sample_coords = torch.stack([lower_coords, upper_coords], dim=0)
    # here we invert the order from upper to lower instead
    neighbor_coords = sample_coords[1 - combinations, ..., selection]
    neighbor_coords = neighbor_coords.permute(2, 3, 0, 1)

    # distances from points to neighbors
    neighbor_dist = torch.abs(query_points.unsqueeze(-2) - neighbor_coords)
    # scale distances to use cubes as a metric
    neighbor_dist /= voxelsize

    # compute trilinear weights for all 8 (if dim=3) neighbors of each sample as:
    # n0 = |x_{n0} - x|/sx * |y_{n0} - y|/sy * |z_{n0} - z|/sz * (feature_value)
    weights = torch.prod(neighbor_dist, dim=-1, keepdim=False)

    # initialize batch and feature indices
    batch_index = torch.arange(batchsize, device=device).expand(samplesize, 2**dim, batchsize)
    batch_index = batch_index.permute(2, 0, 1)

    # get neighbor values
    # NOTE: if python version is 3.11 or higher this if can be removed and
    # value unpacking in subscripts was added
    if dim == 2:
        x, y = tuple(sample_index[..., i] for i in range(dim))
        neighbor_values = field_values[batch_index, x, y]
    elif dim == 3:
        x, y, z = tuple(sample_index[..., i] for i in range(dim))
        neighbor_values = field_values[batch_index, x, y, z]
    else:
        raise NotImplementedError("dim > 3 until transition to python 3.11")

    # weighted sum over neighbor values
    query_values = torch.sum(weights.unsqueeze(-1) * neighbor_values, dim=-2)

    return query_values


class DPSR(torch.nn.Module):
    """Differentiable Poisson Surface Reconstruction
    """
    def __init__(
        self, *, grid: torch.Tensor = torch.tensor([256, 256, 256]), 
        sigma: Number = 5, m: Number = 0.5, eps: float = 1e-6, device = "cpu"
    ):
        """
        Parameters
        ----------
        grid : torch.Tensor, optional
            grid resolution, shape = [dim], by default [256, 256, 256]
        sigma : Number, optional
            smoothing sigma, by default 5
        m : Number, optional
            indicator m, by default 0.5
        eps : float, optional
            division by zero, by default 1e-6
        """
        super().__init__()
        self.grid = grid.to(device)
        self.sigma = sigma
        self.m = m
        self.eps = eps
        # pre-compute vector u
        self.u = get_fft_frequencies(self.grid).unsqueeze(0).to(device)
        # pre-compute vector g^~_{\sigma,r}(u)
        self.g = get_gaussian_smoothing(self.u, self.sigma, self.grid[0]).to(device)

    def forward(self, points: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
        """Approximates the indicator function of an oriented point cloud.
        Function is approximated using a finite grid of specified dimension.

        Parameters
        ----------
        points : torch.Tensor
            a sample of points (shape = [batch, npoints, 3])
            currently all values must be within (0,1)
        normals : torch.Tensor
            normals of said points (shape = [batch, npoints, 3])

        Returns
        -------
        torch.Tensor
            approximation of the indicator function
            shape = [batch, *grid]
        """
        u, g = self.u, self.g

        # compute vector v
        v = point_rasterization(points, normals, self.grid)  # [batch, dim, *grid]
        # compute vector v^~
        v_tilde = torch.fft.fftn(v, dim=(2, 3, 4))
        v_tilde = v_tilde.permute([0, 2, 3, 4, 1])  # [batch, *grid, dim]
        # compute scalar -i/2pi
        _scalar = -1j / (2 * torch.pi)
        # compute vector u @ v^~ / |u|^2
        _vector = torch.sum(u * v_tilde, dim=-1)  # dot-product  # [batch, *grid]
        _vector /= torch.sum(u**2, dim=-1) + self.eps  # [batch, *grid]
        # compute vector \chi^~
        chi_tilde = g * (_scalar * _vector)  # [batch, *grid]
        # compute vector \chi'
        chi_prime = torch.fft.ifftn(chi_tilde, dim=(1, 2, 3))  # [batch, *grid]
        chi_prime = chi_prime.real  # imag is zero

        # compute vector \chi' restricted to x=c
        chi_c = grid_interpolation(
            chi_prime.unsqueeze(-1), points
        ).squeeze(-1)  # [batch, *grid]
        _vector = chi_prime - torch.mean(chi_c, dim=-1)
        # compute scalar \chi' restricted to x=0
        chi_0 = _vector[:, 0, 0, 0]
        _scalar = self.m / torch.abs(chi_0)
        # compute vector \chi
        chi = _scalar * _vector

        return chi 


class MarchingCubes(torch.autograd.Function):
    # TODO: add batchsize > 1 support
    @staticmethod
    def forward(ctx, chi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Given an indicator grid returns vertices, faces, normals.

        Parameters
        ----------
        chi : torch.Tensor
            indicator grid, shape = [1, *grid]

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            vertices, faces, normals
        """
        device = chi.device
        chi_np = chi.squeeze(0).detach().cpu().numpy()
        grid = torch.tensor(chi_np.shape, device=device)

        v, f, n, _ = marching_cubes(chi_np, level=0)
        v = torch.tensor(v.copy(), device=device).unsqueeze(0)
        f = torch.tensor(f.copy(), device=device).unsqueeze(0)
        n = torch.tensor(n.copy(), device=device).unsqueeze(0)
        v /= grid # scale

        ctx.save_for_backward(v, n, grid)
        return v, f, n
    
    @staticmethod
    def backward(ctx, dv: torch.Tensor, df: torch.Tensor, dn: torch.Tensor) -> torch.Tensor:
        """Given gradient (of the loss) with respect to vertices, faces and normals
        returns the gradient with respect to the indicator grid

        Parameters
        ----------
        dv : torch.Tensor
            gradient with respect to vertices
        df : torch.Tensor
            gradient with respect to faces (not used)
        dn : torch.Tensor
            gradient with respect to normals (not used)

        Returns
        -------
        torch.Tensor
            gradient with respect to indicator grid, shape = [1, *grid]
        """
        v, n, grid = ctx.saved_tensors
        # (nvertices, 1, dim) x (nvertices, dim, 1) -> (nvertices, 1, 1)
        # approximating with -n is wrong... maybe older version of skimage
        # used to return the opposite direction?
        di = torch.matmul(dv.permute(1, 0, 2), n.permute(1, 2, 0))
        # approximate indicator grid with sample values (di) in vertices
        di = point_rasterization(v, di.permute(1, 0, 2), grid)
        return di


class ShapeAsPoints:
    """Shape as Points optimization approach
    """
    def __init__(
        self,
        init_points: torch.Tensor, 
        init_normals: torch.Tensor,
        target_pointcloud: torch.Tensor,
        *,
        preprocessing: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        resolution: int = 256,
        sigma: Number = 5,
        eps: float = 1e-6,
        m: float = 0.5,
        device= "cpu"
    ):
        """initialize a new trainer object

        Parameters
        ----------
        init_points : torch.Tensor
            initial point cloud vertices
        init_normals : torch.Tensor
            initial point cloud normals
        target_pointcloud : torch.Tensor
            target point cloud vertices
        preprocessing : Optional[Callable[[torch.Tensor], torch.Tensor]]
            preprocessing function, by default None (unitsphere scaling)
            output of this must be within (0,1) along every dimension
        optimizer: torch.optim.Optimizer
            optimizer used for learning, by default torch.optim.Adam
        resolution : int, optional
            grid resolution (used in DPSR), by default 256
        sigma : Number, optional
            smoothing sigma (used in DPSR), by default 5
        eps : float, optional
            division by zero correction (used in DPSR), by default 1e-6
        m : float, optional
            indicator scaling (used in DPSR), by default 0.5
        """
        dim = init_points.shape[-1]
        self.dpsr = DPSR(
            grid=torch.tensor([resolution] * dim),
            sigma=sigma,
            m=m,
            eps=eps,
            device=device
        ).to(device)
        self.marching_cubes = MarchingCubes.apply
        self.loss = chamfer_distance
        if preprocessing is not None:
            self._points = preprocessing(init_points)
        else:
            self._points = unitsphere_scaling(init_points)
        self._points = self._points.clone().to(device)
        self._points.requires_grad_(True)
        self._normals = init_normals.clone().to(device)
        self._normals.requires_grad_(True)
        self._target = target_pointcloud.clone().to(device)
        
        self.optimizer = optimizer([self._points, self._normals])

        self._t = 0
    
    def step(self, callback: Optional[Callable[[dict], None]] = None):
        """Computes a single step of the optimization method

        Parameters
        ----------
        callback : Optional[Callable[[dict], None]], optional
            callback for plotting/saving, by default None
            dict contains:
            - chi : indicator grid
            - mesh : (v,f,n) of the mesh
            - loss : loss of this step
        """
        self._t += 1
        points = torch.sigmoid(self._points)
        normals = self._normals / self._normals.norm(dim=-1, keepdim=True)

        chi = self.dpsr(points, normals)
        chi = torch.tanh(chi)
        v,f,n = self.marching_cubes(chi)

        loss, _ = self.loss(
            minmax_scaling(v),
            minmax_scaling(self._target),
        )

        loss.backward()

        if callback is not None:
            with torch.no_grad():
                callback({
                    "epoch" : self._t,
                    "chi" : chi,
                    "mesh": (v,f,n),
                    "loss": loss.item()
                })
        
        self.optimizer.step()
        self.optimizer.zero_grad()
    
    def update_resolution(self, res: Number):
        """Update grid resolution.

        Parameters
        ----------
        res : Number
            grid resolution
        """
        self.dpsr = DPSR(
            grid=torch.tensor([res] * self._points.shape[-1]),
            sigma=self.dpsr.sigma,
            m=self.dpsr.m,
            eps=self.dpsr.eps,
            device=self._points.device
        ).to(self._points.device)

    def train(self, schema: Sequence[Tuple[int, int]], *, callback: Optional[Callable[[dict], None]] = None):
        """Execute multiple optimization steps following given schema.
        e.g. if schema = [(32, 100), (128, 200)] then 100 steps at 32 resolution
        followed by 200 steps at 128 resolution will be executed.

        Parameters
        ----------
        schema : Sequence[Tuple[int, int]]
            list of (resolution, iterations)

        callback : Optional[Callable[[dict], None]], optional
            callback passed to self.step (see DPSR.step for reference), by default None
        """
        for res, it in schema:
            self.update_resolution(res)
            for i in range(it):
                self.step(callback)