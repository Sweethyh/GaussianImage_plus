from typing import Callable


def _make_lazy_cuda_func(name: str) -> Callable:
    def call_cuda(*args, **kwargs):
        # pylint: disable=import-outside-toplevel
        from ._backend import _C

        return getattr(_C, name)(*args, **kwargs)

    return call_cuda


nd_rasterize_forward = _make_lazy_cuda_func("nd_rasterize_forward")
nd_rasterize_backward = _make_lazy_cuda_func("nd_rasterize_backward")

nd_rasterize_sum_forward = _make_lazy_cuda_func("nd_rasterize_sum_forward")
nd_rasterize_sum_backward = _make_lazy_cuda_func("nd_rasterize_sum_backward")
nd_rasterize_gs_sum_forward = _make_lazy_cuda_func("nd_rasterize_gs_sum_forward")
nd_rasterize_gs_sum_backward = _make_lazy_cuda_func("nd_rasterize_gs_sum_backward")

rasterize_forward = _make_lazy_cuda_func("rasterize_forward")
rasterize_backward = _make_lazy_cuda_func("rasterize_backward")

compute_cov2d_bounds = _make_lazy_cuda_func("compute_cov2d_bounds")
# compute_cov2d_bounds_xy = _make_lazy_cuda_func("compute_cov2d_bounds_xy")

project_gaussians_forward = _make_lazy_cuda_func("project_gaussians_forward")
project_gaussians_backward = _make_lazy_cuda_func("project_gaussians_backward")

project_gaussians_2d_forward = _make_lazy_cuda_func("project_gaussians_2d_forward")
project_gaussians_2d_backward = _make_lazy_cuda_func("project_gaussians_2d_backward")
# ours
# project_gaussians_plus_forward = _make_lazy_cuda_func("project_gaussians_plus_forward")
# project_gaussians_plus_backward = _make_lazy_cuda_func("project_gaussians_plus_backward")
# project_gaussians_2d_scale_rot_Norm_forward = _make_lazy_cuda_func("project_gaussians_2d_scale_rot_Norm_forward")
# project_gaussians_2d_scale_rot_Norm_backward = _make_lazy_cuda_func("project_gaussians_2d_scale_rot_Norm_backward")
project_gaussians_2d_scale_rot_forward = _make_lazy_cuda_func("project_gaussians_2d_scale_rot_forward")
project_gaussians_2d_scale_rot_backward = _make_lazy_cuda_func("project_gaussians_2d_scale_rot_backward")



project_gaussians_2d_covariance_forward = _make_lazy_cuda_func("project_gaussians_2d_covariance_forward")
project_gaussians_2d_covariance_backward = _make_lazy_cuda_func("project_gaussians_2d_covariance_backward")



compute_sh_forward = _make_lazy_cuda_func("compute_sh_forward")
compute_sh_backward = _make_lazy_cuda_func("compute_sh_backward")
map_gaussian_to_intersects = _make_lazy_cuda_func("map_gaussian_to_intersects")

get_tile_bin_edges = _make_lazy_cuda_func("get_tile_bin_edges")

rasterize_sum_forward = _make_lazy_cuda_func("rasterize_sum_forward")
rasterize_sum_backward = _make_lazy_cuda_func("rasterize_sum_backward")





rasterize_sum_plus_forward = _make_lazy_cuda_func("rasterize_sum_plus_forward")
rasterize_sum_plus_backward= _make_lazy_cuda_func("rasterize_sum_plus_backward")


