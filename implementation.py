# python libraries
from numbers import Number
from typing import Optional, Tuple

# external dependencies
import torch

# not implemented modules
from oracle import grid_interp, point_rasterize

def minmax_scaling(points:torch.Tensor, eps=1e-8) -> torch.tensor:
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


def get_fft_frequencies(grid:Tuple[int, int, int]):
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
        freqs.append(torch.fft.fftfreq(res, d=1/res))
    freqs = torch.stack(torch.meshgrid(freqs, indexing="ij"), dim=-1)
    return freqs


def get_gaussian_smoothing(
    input:torch.Tensor, sigma:Number=5, res:Number=None
):
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
    _scalar = -2*(sigma / res)**2
    return torch.exp(_scalar*_vector)


def point_rasterization(
    points:torch.Tensor, normals:torch.Tensor, grid:Tuple[int,int,int]
):
    """Given a sample of points and normals constructs
    a uniformly sampled grid via trilinear interpolation.

    Parameters
    ----------
    points : torch.Tensor
        sample of points
    normals : torch.Tensor
        normals of said points
    grid : Tuple[int,int,int]
        3d grid dimensions

    Returns
    -------
    torch.Tensor
        shape = [1, 3, *grid]
    """
    # TODO: implement
    # NOTE: implementation below requires points in (0,1)
    # so authors use a sigmoid activation on points
    return point_rasterize(points, normals, grid)



def grid_interpolation(grid_of_values:torch.Tensor, points:torch.Tensor):
    """Given a function approximated by a grid of values, sampled uniformly
    along each dimension (e.g. (32,32,32)), approximate for each point
    given the value the function assumes in said point, using trilinear
    interpolation.

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



def DPSR_forward(self, points:torch.Tensor, normals:torch.Tensor):
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
    v = point_rasterization(points, normals, self.grid)         # [1, 3, 32, 32, 32]
    # compute vector v^~
    v_tilde = torch.fft.fftn(v, dim=(2,3,4))
    v_tilde = v_tilde.permute([0, 2, 3, 4, 1])                  # [1, 32, 32, 32, 3]
    # compute vector u
    u = get_fft_frequencies(self.grid).unsqueeze(0)             # [1, 32, 32, 32, 3]
    # compute vector g^~_{\sigma,r}(u)
    g = get_gaussian_smoothing(u, self.sigma, self.grid[0])     # [1, 32, 32, 32]
    # compute scalar -i/2pi
    _scalar = -1j / (2*torch.pi)
    # compute vector u @ v^~ / |u|^2
    _vector = torch.sum(u*v_tilde, dim=-1)      # dot-product   # [1, 32, 32, 32]
    _vector /= torch.sum(u**2, dim=-1)+self.eps                 # [1, 32, 32, 32]
    # compute vector \chi^~
    chi_tilde = g * (_scalar * _vector)                         # [1, 32, 32, 32]
    # compute vector \chi'
    chi_prime = torch.fft.ifftn(chi_tilde, dim=(1,2,3))         # [1, 32, 32, 32]
    chi_prime = chi_prime.real                  # imag is zero

    # compute vector \chi' restricted to x=c
    chi_c = grid_interpolation(chi_prime, points)
    _vector = chi_prime - torch.mean(chi_c, dim=-1)
    # compute scalar \chi' restricted to x=0
    chi_0 = _vector[:,0,0,0]
    _scalar = self.m / torch.abs(chi_0)
    # compute vector \chi
    chi = _scalar * _vector

    return chi