#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

__global__ void project_gaussians_2d_forward_kernel(
    const int num_points,
    const float clip_coe,
    const float2* __restrict__ means2d,
    const float3* __restrict__ L_elements,
    const dim3 img_size,
    const dim3 tile_bounds,
    const float clip_thresh,
    float2* __restrict__ xys,
    float* __restrict__ depths,
    int* __restrict__ radii,
    float3* __restrict__ conics,
    int32_t* __restrict__ num_tiles_hit,
    float radius_clip,
    bool isprint
);
__global__ void project_gaussians_plus_forward_kernel(
    const int num_points,
    const float clip_coe,
    const float2* __restrict__ means2d,
    const float3* __restrict__ L_elements,
    // const float* __restrict__  opacities,//
    const dim3 img_size,
    const dim3 tile_bounds,
    float2* __restrict__ xys,
    // float* __restrict__ depths,
    int* __restrict__ radii,
    float3* __restrict__ conics,
    int32_t* __restrict__ num_tiles_hit
);
__global__ void project_gaussians_2d_scale_rot_forward_kernel(
    const int num_points,
    const float clip_coe,
    const float2* __restrict__ means2d,
    const float2* __restrict__ scales2d,
    const float* __restrict__ rotation,
    const dim3 img_size,
    const dim3 tile_bounds,
    const float clip_thresh,
    float2* __restrict__ xys,
    float* __restrict__ depths,
    int* __restrict__ radii,
    float3* __restrict__ conics,
    int32_t* __restrict__ num_tiles_hit,
    float radius_clip,
    bool isprint
);
__global__ void project_gaussians_2d_scale_rot_Norm_forward_kernel(
    const int num_points,
    const float clip_coe,
    const float2* __restrict__ means2d,
    const float2* __restrict__ scales2d,
    const float* __restrict__ rotation,
    const dim3 img_size,
    const dim3 tile_bounds,
    const float clip_thresh,
    float2* __restrict__ xys,
    float* __restrict__ depths,
    int* __restrict__ radii,
    float3* __restrict__ conics,
    int32_t* __restrict__ num_tiles_hit,
    float radius_clip,
    bool isprint
);
//ltt
__global__ void project_gaussians_2d_covariance_forward_kernel(
    const int num_points,
    const float clip_coe,
    const float2* __restrict__ means2d,
    const float3* __restrict__ L_elements,
    const dim3 img_size,
    const dim3 tile_bounds,
    const float clip_thresh,
    float2* __restrict__ xys,
    float* __restrict__ depths,
    int* __restrict__ radii,
    float3* __restrict__ conics,
    int32_t* __restrict__ num_tiles_hit,
    float radius_clip,
    bool isprint
);
__global__ void project_gaussians_2d_covariance_xy_forward_kernel(
    const int num_points,
    const float clip_coe,
    const float2* __restrict__ means2d,
    const float3* __restrict__ L_elements,
    const float* __restrict__ opacities,
    const dim3 img_size,
    const dim3 tile_bounds,
    const float clip_thresh,
    // const float opacity_thresh,
    float2* __restrict__ xys,
    float* __restrict__ depths,
    float2* __restrict__ radii,// 2个半径
    float3* __restrict__ conics,
    int32_t* __restrict__ num_tiles_hit,
    bool isprint
) ;
__global__ void project_gaussians_2d_covariance_rd_forward_kernel(
    const int num_points,
    const float clip_coe,
    const float2* __restrict__ means2d,
    const float3* __restrict__ L_elements,
    const float* __restrict__ opacities,
    const dim3 img_size,
    const dim3 tile_bounds,
    const float clip_thresh,
    // const float opacity_thresh,
    float2* __restrict__ xys,
    float* __restrict__ depths,
    int* __restrict__ radii,
    float3* __restrict__ conics,
    int32_t* __restrict__ num_tiles_hit,
    bool isprint
);
__global__ void project_gaussians_2d_covariance_aab_forward_kernel(
    const int num_points,
    const float clip_coe,
    const float2* __restrict__ means2d,
    const float3* __restrict__ L_elements,
    const float* __restrict__ opacities,
    const dim3 img_size,
    const dim3 tile_bounds,
    const float clip_thresh,
    // const float opacity_thresh,
    float2* __restrict__ xys,
    float* __restrict__ depths,
    float2* __restrict__ radii,
    float3* __restrict__ conics,
    int32_t* __restrict__ num_tiles_hit,
    bool isprint
);
// //add object attributes
// __global__ void project_gaussians_2d_object_forward_kernel(
//     const int num_points,
//     const float2* __restrict__ means2d,
//     const float3* __restrict__ L_elements,
//     const float* objects,
//     const dim3 img_size,
//     const dim3 tile_bounds,
//     const float clip_thresh,
//     float2* __restrict__ xys,
//     float* __restrict__ depths,
//     int* __restrict__ radii,
//     float3* __restrict__ conics,
//     int32_t* __restrict__ num_tiles_hit,
//     bool isprint
// );
__global__ void project_gaussians_2d_covariance_norm_forward_kernel(
    const int num_points,
    const float clip_coe,
    const float2* __restrict__ means2d,
    const float3* __restrict__ L_elements,
    // const float* __restrict__ opacities,//
    const dim3 img_size,
    const dim3 tile_bounds,
    const float clip_thresh,// low opacity threshold
    float2* __restrict__ xys,
    float* __restrict__ depths,
    int* __restrict__ radii,
    float3* __restrict__ conics,
    int32_t* __restrict__ num_tiles_hit,
    float radius_clip,
    bool isprint
);
__global__ void project_gaussians_2d_covariance_norm_oob_forward_kernel(
    const int num_points,
    const float clip_coe,
    const float2* __restrict__ means2d,
    const float3* __restrict__ L_elements,
    // const float* __restrict__ opacities,//
    const dim3 img_size,
    const dim3 tile_bounds,
    const float clip_thresh,// low opacity threshold
    float2* __restrict__ xys,
    float* __restrict__ depths,
    float2* __restrict__ radii,// 2个半径
    float3* __restrict__ conics,
    int32_t* __restrict__ num_tiles_hit,
    float2* __restrict__ pix_min_bound,// 
    float2* __restrict__ pix_max_bound,// 
    float radius_clip,
    bool xy_norm,
    bool isprint
);
__global__ void project_gaussians_2d_covariance_norm_oob_intersect_forward_kernel(
    const int num_points,
    const float clip_coe,
    const float2* __restrict__ means2d,
    const float3* __restrict__ L_elements,
    // const float* __restrict__ opacities,//
    const dim3 img_size,
    const dim3 tile_bounds,
    const float clip_thresh,// low opacity threshold
    float2* __restrict__ xys,
    float* __restrict__ depths,
    float2* __restrict__ radii,// 2个半径
    float3* __restrict__ conics,
    int32_t* __restrict__ num_tiles_hit,
    float2 * vertices1,//right up
    float2 * vertices2,//left up
    float2 * vertices3,//right down
    float2 * vertices4,//left down
    bool xy_norm,
    bool isprint
);