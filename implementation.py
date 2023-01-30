# python libraries
from numbers import Number
from typing import Optional, Tuple

# external dependencies
import torch

# not implemented modules
from oracle import grid_interp, point_rasterize


def minmax_scaling(points: torch.Tensor, eps=1e-8) -> torch.Tensor:
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

def centerzoom_scaling(points:torch.Tensor, scale=1.0) -> torch.Tensor:
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
    zoomed = shifted / (scale*scaleto1)
    return zoomed


def get_fft_frequencies(grid: Tuple[int, int, int]) -> torch.Tensor:
    """Returns FFT frequencies for a given grid.

    Parameters
    ----------
    grid : Tuple[int, int, int]
        grid dimensions eg. (32, 32, 32) for a 3-dimensional grid with 32 samples
        along each dimension

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
    input: torch.Tensor, sigma: Number = 5, res: Number = None
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
    points: torch.Tensor, features: torch.Tensor, grid: Tuple[int, int, int]
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
    grid : Tuple[int, int, int]
        field dimension (grid size)

    Returns
    -------
    torch.Tensor
        scalar/vector field of shape [batch, nfeatures, *grid]
    """
    batchsize = points.shape[0]
    samplesize = points.shape[1]
    dim = points.shape[2]
    featuresize = features.shape[2]
    # x0, y0, z0
    # gridstart = torch.tensor([0.]*dim)
    voxelcount = torch.tensor(grid)
    # s0, s1, s2
    voxelsize = 1 / voxelcount
    eps = 1e-5

    # compute neighbor indices
    lower_index = torch.floor(points / voxelsize).int()
    outliers = torch.cat([
        torch.argwhere(lower_index > voxelcount - eps),
        torch.argwhere(lower_index < -eps)
    ])
    if len(outliers) > 0:
        raise ValueError(f"{len(outliers)} points are outside grid limits")
    # lower_index = lower_index.remainder(voxelcount)
    
    upper_index = torch.ceil(points / voxelsize).int()
    outliers = torch.cat([
        torch.argwhere(upper_index > voxelcount + 1 - eps),
        torch.argwhere(upper_index < -eps)
    ])
    if len(outliers) > 0:
        raise ValueError(f"{len(outliers)} points are outside grid limits")
    upper_index = upper_index.remainder(voxelcount)

    sample_index = torch.stack([lower_index, upper_index], dim=0)

    # all subsets of 2**dim elements
    # e.g. if dim=3 : 000, 001, 010, ..., 110, 111
    combinations = torch.stack(
        torch.meshgrid(
            *([torch.tensor([0,1])] * dim), 
            indexing="ij"
        ), 
        dim=-1
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
    sample_index = neighbor_index.permute(2, 3, 0, 1) # [batch, npoints, 2**dim, dim]

    # similarly we construct the cube coordinates
    lower_coords = lower_index * voxelsize
    upper_coords = (lower_index+1) * voxelsize
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
    batch_index = torch.arange(batchsize).expand(samplesize, 2**dim, batchsize)
    batch_index = batch_index.permute(2, 0, 1)
    feature_index = torch.arange(featuresize).reshape(1, 1, 1, featuresize, 1)
    # solve broadcasting issues
    batch_index = batch_index.unsqueeze(-1).unsqueeze(-1)
    batch_index = batch_index.expand(batchsize, samplesize, 2**dim, featuresize, 1)
    sample_index = sample_index.unsqueeze(-2)
    sample_index = sample_index.expand(batchsize, samplesize, 2**dim, featuresize, dim)
    feature_index = feature_index.expand(batchsize, samplesize, 2**dim, featuresize, 1)
    # construct final index
    index = torch.cat([batch_index, feature_index, sample_index], dim=-1)

    # flatten all dimensions
    index = index.reshape(-1, dim+2)
    field_values = field_values.reshape(-1)

    # construct output grid
    output_size = torch.Size((batchsize, featuresize, *grid))
    output_grid = torch.zeros(output_size, dtype=field_values.dtype).view(-1)

    # flatten the index
    index_folds = torch.tensor([1] + list(output_size[:0:-1])).cumprod(0).flip(0)
    index_flat = torch.sum(index * index_folds, dim=-1)

    # write field values to grid at index position
    output_grid.scatter_add_(0, index_flat, field_values)
    output_grid = output_grid.view(*output_size)

    return output_grid


def grid_interpolation(
    grid_of_values: torch.Tensor, points: torch.Tensor
) -> torch.Tensor:
    """Given a function approximated by a grid of values, sampled uniformly
    along each dimension (e.g. (32,32,32)). Approximates, for each point
    given, the value the function assumes in said point.
    The approximation is done via trilinear interpolation.

    Parameters
    ----------
    grid_of_values : torch.Tensor
        shape = [1, *grid]
    points : torch.Tensor
        shape = [1, npoints, len(grid)]

    Returns
    -------
    torch.Tensor
        shape = [1, npoints]
    """
    # TODO: implement
    return grid_interp(grid_of_values.unsqueeze(-1), points).squeeze(-1)


def DPSR_forward(self, points: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
    """Approximates the indicator function of an oriented point cloud.
    Function is approximated using a finite grid of specified dimension.

    Parameters
    ----------
    points : torch.Tensor
        a sample of points (shape = [1, npoints, 3])
        currently all values must be within (0,1)
    normals : torch.Tensor
        normals of said points (shape = [1, npoints, 3])

    Returns
    -------
    torch.Tensor
        approximation of the indicator function
        shape = [1, *grid]
    """
    ### NOTE: I'm not 100% sure but I believe this implementation requires
    ### points to be in (0,1) given how "\chi' restricted to x=0" is computed

    # compute vector v
    v = point_rasterization(points, normals, self.grid)  # [1, 3, 32, 32, 32]
    # compute vector v^~
    v_tilde = torch.fft.fftn(v, dim=(2, 3, 4))
    v_tilde = v_tilde.permute([0, 2, 3, 4, 1])  # [1, 32, 32, 32, 3]
    # compute vector u
    u = get_fft_frequencies(self.grid).unsqueeze(0)  # [1, 32, 32, 32, 3]
    # compute vector g^~_{\sigma,r}(u)
    g = get_gaussian_smoothing(u, self.sigma, self.grid[0])  # [1, 32, 32, 32]
    # compute scalar -i/2pi
    _scalar = -1j / (2 * torch.pi)
    # compute vector u @ v^~ / |u|^2
    _vector = torch.sum(u * v_tilde, dim=-1)  # dot-product  # [1, 32, 32, 32]
    _vector /= torch.sum(u**2, dim=-1) + self.eps  # [1, 32, 32, 32]
    # compute vector \chi^~
    chi_tilde = g * (_scalar * _vector)  # [1, 32, 32, 32]
    # compute vector \chi'
    chi_prime = torch.fft.ifftn(chi_tilde, dim=(1, 2, 3))  # [1, 32, 32, 32]
    chi_prime = chi_prime.real  # imag is zero

    # compute vector \chi' restricted to x=c
    chi_c = grid_interpolation(chi_prime, points)
    _vector = chi_prime - torch.mean(chi_c, dim=-1)
    # compute scalar \chi' restricted to x=0
    chi_0 = _vector[:, 0, 0, 0]
    _scalar = self.m / torch.abs(chi_0)
    # compute vector \chi
    chi = _scalar * _vector

    return chi
