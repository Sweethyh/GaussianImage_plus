"""Python bindings for binning and sorting gaussians"""

from typing import Tuple

from jaxtyping import Float, Int
from torch import Tensor
import torch

import gsplat.cuda as _C


def map_gaussian_to_intersects(
    num_points: int,
    num_intersects: int,
    xys: Float[Tensor, "batch 2"],
    depths: Float[Tensor, "batch 1"],
    radii: Float[Tensor, "batch 1"],
    cum_tiles_hit: Float[Tensor, "batch 1"],
    tile_bounds: Tuple[int, int, int],
    radius_clip: Float=1.0,
    isprint: bool = False,
) -> Tuple[Float[Tensor, "cum_tiles_hit 1"], Float[Tensor, "cum_tiles_hit 1"]]:
    """Map each gaussian intersection to a unique tile ID and depth value for sorting.

    Note:
        This function is not differentiable to any input.

    Args:
        num_points (int): number of gaussians.
        num_intersects (int): total number of tile intersections.
        xys (Tensor): x,y locations of 2D gaussian projections.
        depths (Tensor): z depth of gaussians.
        radii (Tensor): radii of 2D gaussian projections.
        cum_tiles_hit (Tensor): list of cumulative tiles hit.
        tile_bounds (Tuple): tile dimensions as a len 3 tuple (tiles.x , tiles.y, 1).

    Returns:
        A tuple of {Tensor, Tensor}:

        - **isect_ids** (Tensor): unique IDs for each gaussian in the form (tile | depth id).
        - **gaussian_ids** (Tensor): Tensor that maps isect_ids back to cum_tiles_hit.
    """
    # min_r,max_r=torch.min(radii),torch.max(radii)
  
    isect_ids, gaussian_ids = _C.map_gaussian_to_intersects(
        num_points,
        num_intersects,
        xys.contiguous(),
        depths.contiguous(),
        radii.contiguous(),
        cum_tiles_hit.contiguous(),
        tile_bounds,
        radius_clip,
        isprint
        # True
    )
    return (isect_ids, gaussian_ids)
def map_gaussian_to_intersects_plus(
    num_points: int,
    num_intersects: int,
    xys: Float[Tensor, "batch 2"],
    radii: Float[Tensor, "batch 1"],
    cum_tiles_hit: Float[Tensor, "batch 1"],
    tile_bounds: Tuple[int, int, int],
    radius_clip: Float=1.0
) -> Tuple[Float[Tensor, "cum_tiles_hit 1"], Float[Tensor, "cum_tiles_hit 1"]]:
    # min_r,max_r=torch.min(radii),torch.max(radii)
  
    isect_ids, gaussian_ids = _C.map_gaussian_to_intersects_plus(
        num_points,
        num_intersects,
        xys.contiguous(),
        radii.contiguous(),
        cum_tiles_hit.contiguous(),
        tile_bounds,
        radius_clip
    )
    return (isect_ids, gaussian_ids)
def map_gaussian_to_intersects_XY(
    num_points: int,
    num_intersects: int,
    xys: Float[Tensor, "batch 2"],
    depths: Float[Tensor, "batch 1"],
    radii: Float[Tensor, "batch 2"],
    cum_tiles_hit: Float[Tensor, "batch 1"],
     pixels_min_bound: Float[Tensor, "batch 2"],
    pixels_max_bound: Float[Tensor, "batch 2"],
    tile_bounds: Tuple[int, int, int],
    radius_clip: Float=1.0,
     isobb: bool=True,
    isprint: bool = False,
) -> Tuple[Float[Tensor, "cum_tiles_hit 1"], Float[Tensor, "cum_tiles_hit 1"]]:
    """Map each gaussian intersection to a unique tile ID and depth value for sorting.

    Note:
        This function is not differentiable to any input.

    Args:
        num_points (int): number of gaussians.
        num_intersects (int): total number of tile intersections.
        xys (Tensor): x,y locations of 2D gaussian projections.
        depths (Tensor): z depth of gaussians.
        radii (Tensor): radii of 2D gaussian projections.
        cum_tiles_hit (Tensor): list of cumulative tiles hit.
        tile_bounds (Tuple): tile dimensions as a len 3 tuple (tiles.x , tiles.y, 1).

    Returns:
        A tuple of {Tensor, Tensor}:

        - **isect_ids** (Tensor): unique IDs for each gaussian in the form (tile | depth id).
        - **gaussian_ids** (Tensor): Tensor that maps isect_ids back to cum_tiles_hit.
    """
    isect_ids, gaussian_ids = _C.map_gaussian_to_intersects_XY(
        num_points,
        num_intersects,
        xys.contiguous(),
        depths.contiguous(),
        radii.contiguous(),
        cum_tiles_hit.contiguous(),
        pixels_min_bound.contiguous(),
        pixels_max_bound.contiguous(),
        tile_bounds,
        radius_clip,
        isobb,
        isprint
    )
    return (isect_ids, gaussian_ids)
def map_gaussian_to_intersects_OBB(
    num_points: int,
    num_intersects: int,
    xys: Float[Tensor, "batch 2"],
    depths: Float[Tensor, "batch 1"],
    radii: Float[Tensor, "batch 2"],
    cum_tiles_hit: Float[Tensor, "batch 1"],
    tile_bounds: Tuple[int, int, int],
    v: Float[Tensor, "*batch 4 2"], #0-w 0-h,
    # v2: Float[Tensor, "*batch 2"], #0-w 0-h,
    # v3: Float[Tensor, "*batch 2"], #0-w 0-h,
    # v4: Float[Tensor, "*batch 2"], #0-w 0-h,
    radius_clip: Float=1.0,
    isprint: bool = False,
) -> Tuple[Float[Tensor, "cum_tiles_hit 1"], Float[Tensor, "cum_tiles_hit 1"]]:
    
    min_bounds=torch.cat((torch.min(v[:,:,0],dim=1,keepdim=True).values,torch.min(v[:,:,1],dim=1,keepdim=True).values),dim=1)
    max_bounds=torch.cat((torch.max(v[:,:,0],dim=1,keepdim=True).values,torch.max(v[:,:,1],dim=1,keepdim=True).values),dim=1)
    isect_ids, gaussian_ids,real_num_tiles_hit,tile_mask = _C.map_gaussian_to_intersects_OBB(
        num_points,
        num_intersects,
        xys.contiguous(),
        depths.contiguous(),
        radii.contiguous(),
        cum_tiles_hit.contiguous(),
        tile_bounds,
        min_bounds.contiguous(),
        max_bounds.contiguous(),
        v[:,0,:].contiguous(),
        v[:,1,:].contiguous(),
        v[:,2,:].contiguous(),
        v[:,3,:].contiguous(),
        radius_clip,
        isprint
        # True
    )
    return (isect_ids, gaussian_ids,real_num_tiles_hit,tile_mask)

def get_tile_bin_edges(
    num_intersects: int, isect_ids_sorted: Int[Tensor, "num_intersects 1"]
) -> Int[Tensor, "num_intersects 2"]:
    """Map sorted intersection IDs to tile bins which give the range of unique gaussian IDs belonging to each tile.

    Expects that intersection IDs are sorted by increasing tile ID.

    Indexing into tile_bins[tile_idx] returns the range (lower,upper) of gaussian IDs that hit tile_idx.

    Note:
        This function is not differentiable to any input.

    Args:
        num_intersects (int): total number of gaussian intersects.
        isect_ids_sorted (Tensor): sorted unique IDs for each gaussian in the form (tile | depth id).

    Returns:
        A Tensor:

        - **tile_bins** (Tensor): range of gaussians IDs hit per tile.
    """
    return _C.get_tile_bin_edges(num_intersects, isect_ids_sorted.contiguous())


def compute_cov2d_bounds(
    cov2d: Float[Tensor, "batch 3"],clip_coe: float=3.0
) -> Tuple[Float[Tensor, "batch_conics 3"], Float[Tensor, "batch_radii 1"]]:
    """Computes bounds of 2D covariance matrix

    Args:
        cov2d (Tensor): input cov2d of size  (batch, 3) of upper triangular 2D covariance values

    Returns:
        A tuple of {Tensor, Tensor}:

        - **conic** (Tensor): conic parameters for 2D gaussian.
        - **radii** (Tensor): radii of 2D gaussian projections.
    """
    assert (
        cov2d.shape[-1] == 3
    ), f"Expected input cov2d to be of shape (*batch, 3) (upper triangular values), but got {tuple(cov2d.shape)}"
    num_pts = cov2d.shape[0]
    assert num_pts > 0
    return _C.compute_cov2d_bounds(num_pts, clip_coe,cov2d.contiguous())
def compute_cov2d_bounds_xy(
    cov2d: Float[Tensor, "batch 3"],clip_coe: float=3.0
) -> Tuple[Float[Tensor, "batch_conics 3"], Float[Tensor, "batch_radii 2"]]:
    """Computes bounds of 2D covariance matrix

    Args:
        cov2d (Tensor): input cov2d of size  (batch, 3) of upper triangular 2D covariance values

    Returns:
        A tuple of {Tensor, Tensor}:

        - **conic** (Tensor): conic parameters for 2D gaussian.
        - **radii** (Tensor): radii of 2D gaussian projections.
    """
    assert (
        cov2d.shape[-1] == 3
    ), f"Expected input cov2d to be of shape (*batch, 3) (upper triangular values), but got {tuple(cov2d.shape)}"
    num_pts = cov2d.shape[0]
    assert num_pts > 0
    return _C.compute_cov2d_bounds_xy(num_pts, clip_coe,cov2d.contiguous())

def compute_cumulative_intersects(
    num_tiles_hit: Float[Tensor, "batch 1"]
) -> Tuple[int, Float[Tensor, "batch 1"]]:
    """Computes cumulative intersections of gaussians. This is useful for creating unique gaussian IDs and for sorting.

    Note:
        This function is not differentiable to any input.

    Args:
        num_tiles_hit (Tensor): number of intersected tiles per gaussian.

    Returns:
        A tuple of {int, Tensor}:

        - **num_intersects** (int): total number of tile intersections.
        - **cum_tiles_hit** (Tensor): a tensor of cumulated intersections (used for sorting).
    """
    cum_tiles_hit = torch.cumsum(num_tiles_hit, dim=0, dtype=torch.int32)
    num_intersects = cum_tiles_hit[-1].item()
    return num_intersects, cum_tiles_hit


def bin_and_sort_gaussians(
    num_points: int,
    num_intersects: int,
    xys: Float[Tensor, "batch 2"],
    depths: Float[Tensor, "batch 1"],
    radii: Float[Tensor, "batch 1"],
    cum_tiles_hit: Float[Tensor, "batch 1"],
    tile_bounds: Tuple[int, int, int],
    radius_clip: float=1.0,
    isprint: bool = False,
) -> Tuple[
    Float[Tensor, "num_intersects 1"],
    Float[Tensor, "num_intersects 1"],
    Float[Tensor, "num_intersects 1"],
    Float[Tensor, "num_intersects 1"],
    Float[Tensor, "num_intersects 2"],
]:
    """Mapping gaussians to sorted unique intersection IDs and tile bins used for fast rasterization.

    We return both sorted and unsorted versions of intersect IDs and gaussian IDs for testing purposes.

    Note:
        This function is not differentiable to any input.

    Args:
        num_points (int): number of gaussians.
        num_intersects (int): cumulative number of total gaussian intersections
        xys (Tensor): x,y locations of 2D gaussian projections.
        depths (Tensor): z depth of gaussians.
        radii (Tensor): radii of 2D gaussian projections.
        cum_tiles_hit (Tensor): list of cumulative tiles hit.
        tile_bounds (Tuple): tile dimensions as a len 3 tuple (tiles.x , tiles.y, 1).

    Returns:
        A tuple of {Tensor, Tensor, Tensor, Tensor, Tensor}:

        - **isect_ids_unsorted** (Tensor): unique IDs for each gaussian in the form (tile | depth id).
        - **gaussian_ids_unsorted** (Tensor): Tensor that maps isect_ids back to cum_tiles_hit. Useful for identifying gaussians.
        - **isect_ids_sorted** (Tensor): sorted unique IDs for each gaussian in the form (tile | depth id).
        - **gaussian_ids_sorted** (Tensor): sorted Tensor that maps isect_ids back to cum_tiles_hit. Useful for identifying gaussians.
        - **tile_bins** (Tensor): range of gaussians hit per tile.
    """
    # print(torch.cuda.memory_summary())
    isect_ids, gaussian_ids = map_gaussian_to_intersects(
        num_points, num_intersects, xys, depths, radii, cum_tiles_hit, tile_bounds,radius_clip, isprint
    )
    # isect_ids_sorted, sorted_indices = torch.sort(isect_ids)
    try:
        isect_ids_sorted, sorted_indices = torch.sort(isect_ids)
        gaussian_ids_sorted = torch.gather(gaussian_ids, 0, sorted_indices) # 得到一个tile所需要的gs id
    except RuntimeError as e:
        print(f"Caught a RuntimeError: {e}")
        # print(f"isect_ids shape: {isect_ids.shape}, dtype: {isect_ids.dtype}")
        # print(f"sorted_indices shape: {sorted_indices.shape}, dtype: {sorted_indices.dtype}")
        # print(f"gaussian_ids shape: {gaussian_ids.shape}, dtype: {gaussian_ids.dtype}")
        import pdb; pdb.set_trace()
    # gaussian_ids_sorted = torch.gather(gaussian_ids, 0, sorted_indices)
    tile_bins = get_tile_bin_edges(num_intersects, isect_ids_sorted)
    return isect_ids, gaussian_ids, isect_ids_sorted, gaussian_ids_sorted, tile_bins
def bin_and_sort_gaussians_plus(
    num_points: int,
    num_intersects: int,
    xys: Float[Tensor, "batch 2"],
    # depths: Float[Tensor, "batch 1"],
    radii: Float[Tensor, "batch 1"],
    cum_tiles_hit: Float[Tensor, "batch 1"],
    tile_bounds: Tuple[int, int, int],
    radius_clip: float=1.0
    # isprint: bool = False,
) -> Tuple[
    Float[Tensor, "num_intersects 1"],
    Float[Tensor, "num_intersects 1"],
    Float[Tensor, "num_intersects 1"],
    Float[Tensor, "num_intersects 1"],
    Float[Tensor, "num_intersects 2"],
]:
    
    isect_ids, gaussian_ids = map_gaussian_to_intersects_plus(
        num_points, num_intersects, xys, radii, cum_tiles_hit, tile_bounds,radius_clip)
    # isect_ids_sorted, sorted_indices = torch.sort(isect_ids)
    try:
        isect_ids_sorted, sorted_indices = torch.sort(isect_ids)
        gaussian_ids_sorted = torch.gather(gaussian_ids, 0, sorted_indices) # 得到一个tile所需要的gs id
    except RuntimeError as e:
        print(f"Caught a RuntimeError: {e}")
        # print(f"isect_ids shape: {isect_ids.shape}, dtype: {isect_ids.dtype}")
        # print(f"sorted_indices shape: {sorted_indices.shape}, dtype: {sorted_indices.dtype}")
        # print(f"gaussian_ids shape: {gaussian_ids.shape}, dtype: {gaussian_ids.dtype}")
        import pdb; pdb.set_trace()
    # gaussian_ids_sorted = torch.gather(gaussian_ids, 0, sorted_indices)
    tile_bins = get_tile_bin_edges(num_intersects, isect_ids_sorted)
    return isect_ids, gaussian_ids, isect_ids_sorted, gaussian_ids_sorted, tile_bins
def bin_and_sort_gaussians_XY(
    num_points: int,
    num_intersects: int,
    xys: Float[Tensor, "batch 2"],
    depths: Float[Tensor, "batch 1"],
    radii: Float[Tensor, "batch 2"],
    cum_tiles_hit: Float[Tensor, "batch 1"],
    pixels_min_bound: Float[Tensor, "batch 2"],
    pixels_max_bound: Float[Tensor, "batch 2"],
    tile_bounds: Tuple[int, int, int],
    radius_clip: float=1.0,
    isprint: bool = False,
) -> Tuple[
    Float[Tensor, "num_intersects 1"],
    Float[Tensor, "num_intersects 1"],
    Float[Tensor, "num_intersects 1"],
    Float[Tensor, "num_intersects 1"],
    Float[Tensor, "num_intersects 2"],
]:
    isect_ids, gaussian_ids = map_gaussian_to_intersects_XY(
        num_points, num_intersects, xys, depths, radii, cum_tiles_hit,  
        pixels_min_bound, pixels_max_bound,tile_bounds, radius_clip,isprint
    )
    isect_ids_sorted, sorted_indices = torch.sort(isect_ids)
    gaussian_ids_sorted = torch.gather(gaussian_ids, 0, sorted_indices)
    tile_bins = get_tile_bin_edges(num_intersects, isect_ids_sorted)
    return isect_ids, gaussian_ids, isect_ids_sorted, gaussian_ids_sorted, tile_bins
def bin_and_sort_gaussians_OBB(
    num_points: int,
    num_intersects: int,
    xys: Float[Tensor, "batch 2"],
    depths: Float[Tensor, "batch 1"],
    radii: Float[Tensor, "batch 2"],
    cum_tiles_hit: Float[Tensor, "batch 1"],
    tile_bounds: Tuple[int, int, int],
     vertices: Float[Tensor, "batch 4 2"],
    radius_clip: float=1.0,
   
    isprint: bool = False,
) -> Tuple[
    Float[Tensor, "num_intersects 1"],
    Float[Tensor, "num_intersects 1"],
    Float[Tensor, "num_intersects 1"],
    Float[Tensor, "num_intersects 1"],
    Float[Tensor, "num_intersects 2"],
]:
   
    isect_ids, gaussian_ids,real_num_tiles_hit,tile_mask = map_gaussian_to_intersects_OBB(
        num_points, num_intersects, xys, depths, radii, cum_tiles_hit, tile_bounds, vertices, isprint
    )
    # cum_tiles_hit = torch.cumsum(real_num_tiles_hit, dim=0, dtype=torch.int32)
    # num_intersects = cum_tiles_hit[-1].item()
    # isect_ids=isect_ids[tile_mask]
    # gaussian_ids=gaussian_ids[tile_mask]
    isect_ids_sorted, sorted_indices = torch.sort(isect_ids)
    gaussian_ids_sorted = torch.gather(gaussian_ids, 0, sorted_indices)
    # real tile hit nums after apply obb sat
    tile_bins = get_tile_bin_edges(num_intersects, isect_ids_sorted)
    return isect_ids, gaussian_ids, isect_ids_sorted, gaussian_ids_sorted, tile_bins