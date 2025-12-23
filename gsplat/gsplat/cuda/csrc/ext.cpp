#include "bindings.h"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // auto diff functions
    m.def("nd_rasterize_forward", &nd_rasterize_forward_tensor);
    m.def("nd_rasterize_backward", &nd_rasterize_backward_tensor);
    m.def("nd_rasterize_sum_forward", &nd_rasterize_sum_forward_tensor);
    m.def("nd_rasterize_sum_backward", &nd_rasterize_sum_backward_tensor);
    m.def("nd_rasterize_gs_sum_forward", &nd_rasterize_gs_sum_forward_tensor);
    m.def("nd_rasterize_gs_sum_backward", &nd_rasterize_gs_sum_backward_tensor);
    m.def("rasterize_forward", &rasterize_forward_tensor);
    m.def("rasterize_backward", &rasterize_backward_tensor);
    
    //xinjie
    m.def("rasterize_sum_forward", &rasterize_forward_sum_tensor);
    m.def("rasterize_sum_backward", &rasterize_backward_sum_tensor);
    
    
   

    // ours
    m.def("rasterize_sum_plus_forward", &rasterize_sum_plus_forward_tensor);
    m.def("rasterize_sum_plus_backward", &rasterize_sum_plus_backward_tensor);


    m.def("project_gaussians_forward", &project_gaussians_forward_tensor);
    m.def("project_gaussians_backward", &project_gaussians_backward_tensor);
    m.def("compute_sh_forward", &compute_sh_forward_tensor);
    m.def("compute_sh_backward", &compute_sh_backward_tensor);
    m.def("project_gaussians_2d_forward", &project_gaussians_2d_forward_tensor);
    m.def("project_gaussians_2d_backward", &project_gaussians_2d_backward_tensor);
    m.def("project_gaussians_2d_scale_rot_forward", &project_gaussians_2d_scale_rot_forward_tensor);
    m.def("project_gaussians_2d_scale_rot_backward", &project_gaussians_2d_scale_rot_backward_tensor);
    
    
    //ours 
    m.def("project_gaussians_2d_covariance_forward", &project_gaussians_2d_covariance_forward_tensor);
    m.def("project_gaussians_2d_covariance_backward", &project_gaussians_2d_covariance_backward_tensor);
    

    // //ltt 直接优化协方差矩阵 normalized coordinates
    // m.def("project_gaussians_2d_covariance_norm_forward", &project_gaussians_2d_covariance_norm_forward_tensor);
    // m.def("project_gaussians_2d_covariance_norm_backward", &project_gaussians_2d_covariance_norm_backward_tensor);
    // // 不同 的radius 优化
    // m.def("project_gaussians_2d_covariance_xy_forward", &project_gaussians_2d_covariance_xy_forward_tensor);
    // m.def("project_gaussians_2d_covariance_xy_backward", &project_gaussians_2d_covariance_xy_backward_tensor);
    // m.def("project_gaussians_2d_covariance_rd_forward", &project_gaussians_2d_covariance_rd_forward_tensor);
    // // m.def("project_gaussians_2d_covariance_rd_backward", &project_gaussians_2d_covariance_rd_backward_tensor);
    // m.def("project_gaussians_2d_covariance_aab_forward", &project_gaussians_2d_covariance_aab_forward_tensor);
    // m.def("project_gaussians_2d_covariance_aab_backward", &project_gaussians_2d_covariance_aab_backward_tensor);
   
    // m.def("project_gaussians_2d_covariance_obb_forward", &project_gaussians_2d_covariance_OBB_forward_tensor);
    // m.def("project_gaussians_2d_covariance_obb_intersect_forward", &project_gaussians_2d_covariance_OBB_Intersect_forward_tensor);
    // m.def("project_gaussians_2d_covariance_obb_backward", &project_gaussians_2d_covariance_obb_backward_tensor);
    
    // utils
    m.def("compute_cov2d_bounds", &compute_cov2d_bounds_tensor);
    m.def("compute_cov2d_bounds_xy", &compute_cov2d_bounds_tensor);
    m.def("map_gaussian_to_intersects", &map_gaussian_to_intersects_tensor);
    // m.def("map_gaussian_to_intersects_XY", &map_gaussian_to_intersects_tensor_XY);//ltt
    // m.def("map_gaussian_to_intersects_OBB", &map_gaussian_to_intersects_tensor_obb);//ltt
    // m.def("map_gaussian_to_intersects_plus", &map_gaussian_to_intersects_plus_tensor);//ltt


    m.def("get_tile_bin_edges", &get_tile_bin_edges_tensor);

   
}
