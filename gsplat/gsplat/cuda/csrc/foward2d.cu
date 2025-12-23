#include "forward2d.cuh"
#include "helpers.cuh"
#include <algorithm>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <iostream>
namespace cg = cooperative_groups;

// kernel function for projecting each gaussian on device
// each thread processes one gaussian
// 这个函数应该包含的过程：计算2d cov，num_tiles_hit tile_bounds blabla 然后接入rasterize_forward？
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
) {
    unsigned idx = cg::this_grid().thread_rank(); // idx of thread within grid
    if (idx >= num_points) {
        return;
    }
    radii[idx] = 0;
    num_tiles_hit[idx] = 0;
      // 先给一个固定的depth，为了后面的函数调用方便
    depths[idx] = 0.0f;
    // Retrieve the 2D Gaussian parameters
    // printf("means2d %d, %.2f %.2f \n", idx, means2d[idx].x, means2d[idx].y);
    // float clamped_x = max(-1.0f, min(1.0f, means2d[idx].x)); // Clamp x between -1 and 1
    // float clamped_y = max(-1.0f, min(1.0f, means2d[idx].y)); // Clamp y between -1 and 1

    float2 center = {0.5f * img_size.x * means2d[idx].x + 0.5f * img_size.x,
                     0.5f * img_size.y * means2d[idx].y + 0.5f * img_size.y};
    // Assuming L is packed row-wise in a 1D array: [l11, l21, l22]
    float l11 = L_elements[idx].x; // scale_x
    float l21 = L_elements[idx].y; // covariance_xy
    float l22 = L_elements[idx].z; // scale_y

    float3 cov2d = make_float3(l11*l11, l11*l21, l21*l21 + l22*l22);
    // printf("cov2d %d, %.2f %.2f %.2f\n", idx, cov2d.x, cov2d.y, cov2d.z);
    float3 conic;
    float2 radius;
    bool ok = compute_cov2d_bounds(cov2d, conic, radius,clip_coe);
    if (!ok)
        return; // zero determinant
    if (radius.y<radius_clip) //短轴不小于指定阈值， 否则会出现alias
        return;
    conics[idx] = conic;
    xys[idx] = center;
    radii[idx] = (int)radius.x;
    uint2 tile_min, tile_max;
    get_tile_bbox(center, radius.x, tile_bounds, tile_min, tile_max);
    int32_t tile_area = (tile_max.x - tile_min.x) * (tile_max.y - tile_min.y);
    if (tile_area <= 0) {
        if (isprint)
            printf("%d point bbox outside of bounds\n", idx);
        return;
    }
    num_tiles_hit[idx] = tile_area;
}

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
) {
    unsigned idx = cg::this_grid().thread_rank(); // idx of thread within grid
    if (idx >= num_points) {
        return;
    }
    radii[idx] = 0;
    num_tiles_hit[idx] = 0;
     // 先给一个固定的depth，为了后面的函数调用方便
    depths[idx] = 0.0f;
    // Retrieve the 2D Gaussian parameters
    float2 center = {0.5f * img_size.x * means2d[idx].x + 0.5f * img_size.x,
                     0.5f * img_size.y * means2d[idx].y + 0.5f * img_size.y};

    glm::mat2 R = rotmat2d(rotation[idx]);
    glm::mat2 S = scale_to_mat2d(scales2d[idx]);
    glm::mat2 M = R * S;
    glm::mat2 tmp = M * glm::transpose(M);
    // glm::mat2 tmp = R * S * glm::transpose(R);

    float3 cov2d = make_float3(tmp[0][0], tmp[0][1], tmp[1][1]);
    // printf("cov2d %d, %.2f %.2f %.2f\n", idx, cov2d.x, cov2d.y, cov2d.z);
    float3 conic;
    float2 radius;
    bool ok = compute_cov2d_bounds(cov2d, conic, radius,clip_coe);
    if (!ok)
        return; // zero determinant
    if (radius.y<radius_clip) //短轴不小于指定阈值， 否则会出现alias
        return;
    conics[idx] = conic;
    xys[idx] = center;
    radii[idx] = (int)radius.x;
    uint2 tile_min, tile_max;
    get_tile_bbox(center, radii[idx], tile_bounds, tile_min, tile_max);
    int32_t tile_area = (tile_max.x - tile_min.x) * (tile_max.y - tile_min.y);
    if (tile_area <= 0) {
        if (isprint)
            printf("%d point bbox outside of bounds\n", idx);
        return;
    }
    num_tiles_hit[idx] = tile_area;
   

}
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
) {
    unsigned idx = cg::this_grid().thread_rank(); // idx of thread within grid
    if (idx >= num_points) {
        return;
    }
    radii[idx] = 0;
    num_tiles_hit[idx] = 0;
     // 先给一个固定的depth，为了后面的函数调用方便
    depths[idx] = 0.0f;
    // Retrieve the 2D Gaussian parameters
    float2 center = {means2d[idx].x ,means2d[idx].y };

    glm::mat2 R = rotmat2d(rotation[idx]);
    glm::mat2 S = scale_to_mat2d(scales2d[idx]);
    glm::mat2 M = R * S;
    glm::mat2 tmp = M * glm::transpose(M);
    // glm::mat2 tmp = R * S * glm::transpose(R);

    float3 cov2d = make_float3(tmp[0][0], tmp[0][1], tmp[1][1]);
    // printf("cov2d %d, %.2f %.2f %.2f\n", idx, cov2d.x, cov2d.y, cov2d.z);
    float3 conic;
    float2 radius;
    bool ok = compute_cov2d_bounds(cov2d, conic, radius,clip_coe);
    if (!ok)
        return; // zero determinant
    if (radius.y<radius_clip) //短轴不小于指定阈值， 否则会出现alias
        return;
    conics[idx] = conic;
    xys[idx] = center;
    radii[idx] = (int)radius.x;
    uint2 tile_min, tile_max;
    get_tile_bbox(center, radii[idx], tile_bounds, tile_min, tile_max);
    int32_t tile_area = (tile_max.x - tile_min.x) * (tile_max.y - tile_min.y);
    if (tile_area <= 0) {
        if (isprint)
            printf("%d point bbox outside of bounds\n", idx);
        return;
    }
    num_tiles_hit[idx] = tile_area;
   

}
// kernel function for projecting each gaussian on device
// each thread processes one gaussian
//直接优化协方差矩阵 不进行分解  参考LIG
// 这个函数应该包含的过程：计算2d cov，num_tiles_hit tile_bounds blabla 然后接入rasterize_forward？
__global__ void project_gaussians_2d_covariance_forward_kernel(
    const int num_points,
    const float clip_coe,
    const float2* __restrict__ means2d,
    const float3* __restrict__ L_elements,
    // const float* __restrict__  opacities,//
    const dim3 img_size,
    const dim3 tile_bounds,
    const float clip_thresh,// low opacity threshold
    float2* __restrict__ xys,
    float* __restrict__ depths,
    int* __restrict__ radii,
    float3* __restrict__ conics,
    int32_t* __restrict__ num_tiles_hit,
    float radius_clip,
    // float* pixel_sizes,
    bool isprint
) {
    unsigned idx = cg::this_grid().thread_rank(); // idx of thread within grid
    if (idx >= num_points) {//index超过2D gaussian总数的线程直接返回, 防止数组越界访问
        return;
    }
    radii[idx] = 0;
    num_tiles_hit[idx] = 0;
    // 先给一个固定的depth，为了后面的函数调用方便
    depths[idx] = 0.0f;
    // pixel_sizes[idx] = 0;
    // Retrieve the 2D Gaussian parameters
    // printf("means2d %d, %.2f %.2f \n", idx, means2d[idx].x, means2d[idx].y);
    // float clamped_x = max(-1.0f, min(1.0f, means2d[idx].x)); // Clamp x between -1 and 1
    // float clamped_y = max(-1.0f, min(1.0f, means2d[idx].y)); // Clamp y between -1 and 1

    // float2 center = {0.5f * img_size.x * means2d[idx].x + 0.5f * img_size.x,
    //                  0.5f * img_size.y * means2d[idx].y + 0.5f * img_size.y};
    float2 center = {means2d[idx].x ,means2d[idx].y };
    // Assuming L is packed row-wise in a 1D array: [l11, l21, l22]
    float l11 = L_elements[idx].x; // scale_x
    float l21 = L_elements[idx].y; // covariance_xy
    float l22 = L_elements[idx].z; // scale_y

    // Construct the 2x2 covariance matrix from L
    // float2x2 Cov2D = make_float2x2(l11*l11, l11*l21,
                                //    l11*l21, l21*l21 + l22*l22);
    // float3 cov2d = make_float3(l11*l11, l21, l22*l22);
    float3 cov2d = make_float3(l11, l21, l22); // x的方差，cov(x,y),y的方差
    // printf("cov2d %d, %.2f %.2f %.2f\n", idx, cov2d.x, cov2d.y, cov2d.z);

    //计算协方差的逆 和 椭圆半径
    float3 conic;
    float2 radius;
    bool ok = compute_cov2d_bounds(cov2d, conic, radius,clip_coe); 
    
    if (!ok) //cov2d不可逆
        return; // zero determinant
    if (radius.y<radius_clip) //短轴不小于指定阈值， 否则会出现alias
        return;
    if(isprint)
        printf("gs idx %d conic %.2f %.2f %.2f\n", idx, conic.x, conic.y, conic.z);
    // printf("conic %d %.2f %.2f %.2f\n", idx, conic.x, conic.y, conic.z);
    conics[idx] = conic;
    xys[idx] = center;
    radii[idx] = (int)radius.x;

    // // calc pixel size
    // 	// calculate pixel size of the gaussian
	// float occ = opacities[idx];
    // float level_set = -2 * log(1 / (255.0f * occ));
    // level_set = max(0.0f, level_set);    // negative level set when gaussian opacity is too low
    // //    float dx = sqrt(level_set / conic.x);
    // //    float dy = sqrt(level_set / conic.z);
    // float dx = sqrt(level_set / conic.x);
    // float dy = sqrt(level_set / conic.z);
    // float pixel_size = min(dx, dy);
    // // pixel_size /= scale_modifier;       // use original gaussian size for filtering, for more faithful visualization
    // pixel_sizes[idx] = pixel_size;

    // float rel_min_pixel_size = 1.0f;
    // if (min_pixel_sizes[idx] > 0) {
    //     rel_min_pixel_size = pixel_size / min_pixel_sizes[idx];
    // }
    // float rel_max_pixel_size = 1.0f;
    // if (max_pixel_sizes[idx] > 0) {
    //     rel_max_pixel_size = pixel_size / max_pixel_sizes[idx];
    // }
    uint2 tile_min, tile_max;
    // 当前gs落在那几个tile上
    get_tile_bbox(center, radius.x, tile_bounds, tile_min, tile_max);
    int32_t tile_area = (tile_max.x - tile_min.x) * (tile_max.y - tile_min.y);//覆盖了多少个tile
    if (tile_area <= 0) {
        if (isprint)
            printf("%d point bbox outside of bounds\n", idx);
        return;
    }
    num_tiles_hit[idx] = tile_area;
    

}
__global__ void project_gaussians_plus_forward_kernel(
    const int num_points,
    const float clip_coe,
    const float2* __restrict__ means2d,
    const float3* __restrict__ L_elements,
    // const float* __restrict__  opacities,//
    const dim3 img_size,
    const dim3 tile_bounds,
    // const float clip_thresh,// low opacity threshold
    float2* __restrict__ xys,
    // float* __restrict__ depths,
    int* __restrict__ radii,
    float3* __restrict__ conics,
    int32_t* __restrict__ num_tiles_hit
    // float radius_clip,
    // float* pixel_sizes,
    // bool isprint
) {
    unsigned idx = cg::this_grid().thread_rank(); // idx of thread within grid
    if (idx >= num_points) {//index超过2D gaussian总数的线程直接返回, 防止数组越界访问
        return;
    }
    radii[idx] = 0;
    num_tiles_hit[idx] = 0;
    // 先给一个固定的depth，为了后面的函数调用方便
    // depths[idx] = 0.0f;
    float2 center = means2d[idx];
    xys[idx] = center;

    // Assuming L is packed row-wise in a 1D array: [l11, l21, l22]
    // float l11 = L_elements[idx].x; // scale_x
    // float l21 = L_elements[idx].y; // covariance_xy
    // float l22 = L_elements[idx].z; // scale_y

    // float3 cov2d = make_float3(l11, l21, l22); // x的方差，cov(x,y),y的方差
    float3 cov2d=L_elements[idx];
    // printf("cov2d %d, %.2f %.2f %.2f\n", idx, cov2d.x, cov2d.y, cov2d.z);
    //计算协方差的逆 和 椭圆半径
    float3 conic;
    float2 radius;
    bool ok = compute_cov2d_bounds(cov2d, conic, radius,clip_coe); 
    
    if (!ok) //cov2d不可逆
        return; // zero determinant
    conics[idx] = conic;
    
    
    uint2 tile_min, tile_max;
    // 当前gs落在那几个tile上
    get_tile_bbox(center, radius.x, tile_bounds, tile_min, tile_max);
    int32_t tile_area = (tile_max.x - tile_min.x) * (tile_max.y - tile_min.y);//覆盖了多少个tile
    if (tile_area <= 0) {
        return;
    }
    num_tiles_hit[idx] = tile_area;
    radii[idx] = (int)radius.x;
}
// // kernel function for projecting each gaussian on device
// // each thread processes one gaussian
// //直接优化协方差矩阵 不进行分解  参考LIG
// // 这个函数应该包含的过程：计算2d cov，num_tiles_hit tile_bounds blabla 然后接入rasterize_forward？
// __global__ void project_gaussians_2d_covariance_norm_forward_kernel(
//     const int num_points,
//     const float clip_coe,
//     const float2* __restrict__ means2d,
//     const float3* __restrict__ L_elements,
//     // const float* __restrict__ opacities,//
//     const dim3 img_size,
//     const dim3 tile_bounds,
//     const float clip_thresh,// low opacity threshold
//     float2* __restrict__ xys,
//     float* __restrict__ depths,
//     int* __restrict__ radii,
//     float3* __restrict__ conics,
//     int32_t* __restrict__ num_tiles_hit,
//     float radius_clip,
//     bool isprint
// ) {
//     unsigned idx = cg::this_grid().thread_rank(); // idx of thread within grid
//     if (idx >= num_points) {//index超过2D gaussian总数的线程直接返回, 防止数组越界访问
//         return;
//     }
//     radii[idx] = 0;
//     num_tiles_hit[idx] = 0;
//     // 先给一个固定的depth，为了后面的函数调用方便
//     depths[idx] = 0.0f;
//
//     float2 center = {0.5f * img_size.x * means2d[idx].x + 0.5f * img_size.x,
//         0.5f * img_size.y * means2d[idx].y + 0.5f * img_size.y};
//     float l11 = L_elements[idx].x; // scale_x
//     float l21 = L_elements[idx].y; // covariance_xy
//     float l22 = L_elements[idx].z; // scale_y
//     float3 cov2d = make_float3(l11, l21, l22); // x的方差，cov(x,y),y的方差
//     // printf("cov2d %d, %.2f %.2f %.2f\n", idx, cov2d.x, cov2d.y, cov2d.z);
//
//     //计算协方差的逆 和 椭圆半径
//     float3 conic;
//     float2 radius;
//     bool ok = compute_cov2d_bounds(cov2d, conic, radius,clip_coe);
//
//     if (!ok) //cov2d不可逆
//         return; // zero determinant
//     if (radius.y<radius_clip) //短轴不小于指定阈值， 否则会出现alias
//         return;
//     if(isprint)
//         printf("gs idx %d conic %.2f %.2f %.2f\n", idx, conic.x, conic.y, conic.z);
//     // printf("conic %d %.2f %.2f %.2f\n", idx, conic.x, conic.y, conic.z);
//     conics[idx] = conic;
//     xys[idx] = center;
//     radii[idx] = (int)radius.x;
//     uint2 tile_min, tile_max;
//     // 当前gs落在那几个tile上
//     get_tile_bbox(center, radius.x, tile_bounds, tile_min, tile_max);
//     int32_t tile_area = (tile_max.x - tile_min.x) * (tile_max.y - tile_min.y);//覆盖了多少个tile
//     if (tile_area <= 0) {
//         if (isprint)
//             printf("%d point bbox outside of bounds\n", idx);
//         return;
//     }
//     num_tiles_hit[idx] = tile_area;
// }
// __global__ void project_gaussians_2d_covariance_norm_oob_forward_kernel(
//     const int num_points,
//     const float clip_coe,
//     const float2* __restrict__ means2d,
//     const float3* __restrict__ L_elements,
//     // const float* __restrict__ opacities,//
//     const dim3 img_size,
//     const dim3 tile_bounds,
//     const float clip_thresh,// low opacity threshold
//     float2* __restrict__ xys,
//     float* __restrict__ depths,
//     float2* __restrict__ radii,// 2个半径
//     float3* __restrict__ conics,
//     int32_t* __restrict__ num_tiles_hit,
//     float2* __restrict__ pixels_min_bound,//
//     float2* __restrict__ pixels_max_bound,//
//     float radius_clip,
//     bool xy_norm,
//     bool isprint
// ) {
//     unsigned idx = cg::this_grid().thread_rank(); // idx of thread within grid
//     if (idx >= num_points) {//index超过2D gaussian总数的线程直接返回, 防止数组越界访问
//         return;
//     }
//     radii[idx].x=0.f;
//     radii[idx].y=0.f;
//     num_tiles_hit[idx] = 0;
//     pixels_min_bound[idx]={0.f,0.f};
//     pixels_max_bound[idx]={0.f,0.f};
//     // 先给一个固定的depth，为了后面的函数调用方便
//     depths[idx] = 0.0f;
//     float2 center;
//     if  (xy_norm )
//         center={0.5f * img_size.x * means2d[idx].x + 0.5f * img_size.x, 0.5f * img_size.y * means2d[idx].y + 0.5f * img_size.y};
//     else
//         center={means2d[idx].x ,means2d[idx].y };
//     // float2 center = {means2d[idx].x ,means2d[idx].y };
//     // Assuming L is packed row-wise in a 1D array: [l11, l21, l22]
//     float l11 = L_elements[idx].x; // scale_x
//     float l21 = L_elements[idx].y; // covariance_xy
//     float l22 = L_elements[idx].z; // scale_y
//     float3 cov2d = make_float3(l11, l21, l22); // x的方差，cov(x,y),y的方差
//     // printf("cov2d %d, %.2f %.2f %.2f\n", idx, cov2d.x, cov2d.y, cov2d.z);
//
//     //计算协方差的逆 和 椭圆半径
//     float3 conic;
//     float2 radius;
//     float2 vertices[4];
//     bool ok = compute_cov2d_obb_bounds(vertices[0],vertices[1],vertices[2],vertices[3],center, cov2d, conic, radius, clip_coe);
//     if (!ok) //cov2d不可逆
//         return; // zero determinant
//     if (radius.y<radius_clip) //短轴不小于指定阈值， 否则会出现alias
//         return;
//     if(isprint)
//         printf("gs idx %d conic %.2f %.2f %.2f\n", idx, conic.x, conic.y, conic.z);
//     // printf("conic %d %.2f %.2f %.2f\n", idx, conic.x, conic.y, conic.z);
//     conics[idx] = conic;
//     xys[idx] = center;
//     radii[idx] = radius;
//     uint2 tile_min, tile_max;
//     float2 min_bound = vertices[0];
//     float2 max_bound = vertices[0];
//     for (int i = 1; i < 4; ++i) {
//         if (vertices[i].x < min_bound.x) min_bound.x = vertices[i].x;
//         if (vertices[i].x > max_bound.x) max_bound.x = vertices[i].x;
//         if (vertices[i].y < min_bound.y) min_bound.y = vertices[i].y;
//         if (vertices[i].y > max_bound.y) max_bound.y = vertices[i].y; }
//     // 当前gs落在那几个tile上
//     get_tile_obb_bbox(min_bound, max_bound,  tile_bounds,tile_min, tile_max);
//     int32_t tile_area = ( tile_max.x - tile_min.x) * ( tile_max.y - tile_min.y);//覆盖了多少个tile
//     if (tile_area <= 0) {
//         if (isprint)
//             printf("%d point bbox outside of bounds\n", idx);
//         return;
//     }
//     num_tiles_hit[idx] = tile_area;
//     pixels_min_bound[idx]=min_bound;
//     pixels_max_bound[idx]=max_bound;
//
// }
// __global__ void project_gaussians_2d_covariance_norm_oob_intersect_forward_kernel(
//     const int num_points,
//     const float clip_coe,
//     const float2* __restrict__ means2d,
//     const float3* __restrict__ L_elements,
//     // const float* __restrict__ opacities,//
//     const dim3 img_size,
//     const dim3 tile_bounds,
//     const float clip_thresh,// low opacity threshold
//     float2* __restrict__ xys,
//     float* __restrict__ depths,
//     float2* __restrict__ radii,// 2个半径
//     float3* __restrict__ conics,
//     int32_t* __restrict__ num_tiles_hit,
//     float2 * vertices1,//right up
//     float2 * vertices2,//left up
//     float2 * vertices3,//right down
//     float2 * vertices4,//left down
//     bool xy_norm,
//     bool isprint
// ) {
//     unsigned idx = cg::this_grid().thread_rank(); // idx of thread within grid
//     if (idx >= num_points) {//index超过2D gaussian总数的线程直接返回, 防止数组越界访问
//         return;
//     }
//     radii[idx].x=0.f;
//     radii[idx].y=0.f;
//     num_tiles_hit[idx] = 0;
//     // 先给一个固定的depth，为了后面的函数调用方便
//     depths[idx] = 0.0f;
//     float2 center;
//     if (xy_norm )
//         center={0.5f * img_size.x * means2d[idx].x + 0.5f * img_size.x, 0.5f * img_size.y * means2d[idx].y + 0.5f * img_size.y};
//     else
//         center={means2d[idx].x ,means2d[idx].y };
//     // float2 center = {means2d[idx].x ,means2d[idx].y };
//     // Assuming L is packed row-wise in a 1D array: [l11, l21, l22]
//     float l11 = L_elements[idx].x; // scale_x
//     float l21 = L_elements[idx].y; // covariance_xy
//     float l22 = L_elements[idx].z; // scale_y
//     float3 cov2d = make_float3(l11, l21, l22); // x的方差，cov(x,y),y的方差
//     // printf("cov2d %d, %.2f %.2f %.2f\n", idx, cov2d.x, cov2d.y, cov2d.z);
//
//     //计算协方差的逆 和 椭圆半径
//     float3 conic;
//     float2 radius;
//     bool ok = compute_cov2d_obb_bounds(vertices1[idx],vertices2[idx],vertices3[idx],vertices4[idx], center, cov2d, conic, radius, clip_coe);
//     if (!ok) //cov2d不可逆
//         return; // zero determinant
//     if(isprint)
//         printf("gs idx %d conic %.2f %.2f %.2f\n", idx, conic.x, conic.y, conic.z);
//     // printf("conic %d %.2f %.2f %.2f\n", idx, conic.x, conic.y, conic.z);
//     conics[idx] = conic;
//     xys[idx] = center;
//     radii[idx] = radius;
//
//     // 当前gs落在那几个tile上
//
//     // 当前gs obb box的4个顶点 的坐标的最值
//     float2 min_bound = vertices1[idx];
//     float2 max_bound = vertices1[idx];
//     if (vertices2[idx].x < min_bound.x) min_bound.x = vertices2[idx].x;
//     if (vertices2[idx].x > max_bound.x) max_bound.x = vertices2[idx].x;
//     if (vertices2[idx].y < min_bound.y) min_bound.y = vertices2[idx].y;
//     if (vertices2[idx].y > max_bound.y) max_bound.y = vertices2[idx].y;
//
//     if (vertices3[idx].x < min_bound.x) min_bound.x = vertices3[idx].x;
//     if (vertices3[idx].x > max_bound.x) max_bound.x = vertices3[idx].x;
//     if (vertices3[idx].y < min_bound.y) min_bound.y = vertices3[idx].y;
//     if (vertices3[idx].y > max_bound.y) max_bound.y = vertices3[idx].y;
//
//     if (vertices4[idx].x < min_bound.x) min_bound.x = vertices4[idx].x;
//     if (vertices4[idx].x > max_bound.x) max_bound.x = vertices4[idx].x;
//     if (vertices4[idx].y < min_bound.y) min_bound.y = vertices4[idx].y;
//     if (vertices4[idx].y > max_bound.y) max_bound.y = vertices4[idx].y;
//
//     uint2 tile_min, tile_max;
//     get_tile_obb_bbox(min_bound, max_bound,  tile_bounds, tile_min, tile_max);
//     int32_t tile_area = (tile_max.x - tile_min.x) * ( tile_max.y - tile_min.y);//覆盖了多少个tile
//     if (tile_area <= 0) {
//         if (isprint)
//             printf("%d point bbox outside of bounds\n", idx);
//         return;
//     }
//     num_tiles_hit[idx] = tile_area;
// }
// //radius 有2个
// __global__ void project_gaussians_2d_covariance_xy_forward_kernel(
//     const int num_points,
//     const float clip_coe,
//     const float2* __restrict__ means2d,
//     const float3* __restrict__ L_elements,
//     const float* __restrict__ opacities,//
//     const dim3 img_size,
//     const dim3 tile_bounds,
//     const float clip_thresh,
//     float2* __restrict__ xys,
//     float* __restrict__ depths,
//     float2* __restrict__ radii,// 2个半径
//     float3* __restrict__ conics,
//     int32_t* __restrict__ num_tiles_hit,
//     bool isprint
// ) {
//     unsigned idx = cg::this_grid().thread_rank(); // idx of thread within grid
//     if (idx >= num_points) {//index超过2D gaussian总数的线程直接返回, 防止数组越界访问
//         return;
//     }
//     radii[idx].x = 0.0;
//     radii[idx].y= 0.0;
//     num_tiles_hit[idx] = 0;
//     // 先给一个固定的depth，为了后面的函数调用方便
//     depths[idx] = 0.0f;
//     float2 center = {means2d[idx].x ,means2d[idx].y };
//     // Assuming L is packed row-wise in a 1D array: [l11, l21, l22]
//     float l11 = L_elements[idx].x; // scale_x
//     float l21 = L_elements[idx].y; // covariance_xy
//     float l22 = L_elements[idx].z; // scale_y
//
//     float3 cov2d = make_float3(l11, l21, l22); // x的方差，cov(x,y),y的方差
//     // printf("cov2d %d, %.2f %.2f %.2f\n", idx, cov2d.x, cov2d.y, cov2d.z);
//
//     //计算协方差的逆 和 椭圆半径
//     float3 conic;
//     float2 radius;
//     // const float opacity = opacities[idx];
//     // bool ok = compute_cov2d_bounds_xy(cov2d, conic, radius,clip_coe);
//     bool ok = compute_cov2d_bounds_xy(cov2d, conic,radius);
//
//     if (!ok) //cov2d不可逆
//         return; // zero determinant
//     if(isprint)
//         printf("gs idx %d conic %.2f %.2f %.2f\n", idx, conic.x, conic.y, conic.z);
//     // printf("conic %d %.2f %.2f %.2f\n", idx, conic.x, conic.y, conic.z);
//     conics[idx] = conic;
//     xys[idx] = center;
//     radii[idx] = radius;
//     uint2 tile_min, tile_max;
//     // 当前gs落在那几个tile上
//     get_tile_bbox_xy(center, radius, tile_bounds, tile_min, tile_max);
//     int32_t tile_area = (tile_max.x - tile_min.x) * (tile_max.y - tile_min.y);//覆盖了多少个tile
//     if (tile_area <= 0) {
//         if (isprint)
//             printf("%d point bbox outside of bounds\n", idx);
//         return;
//     }
//     num_tiles_hit[idx] = tile_area;
//
//
// }
// //adaptive_radius
// __global__ void project_gaussians_2d_covariance_rd_forward_kernel(
//     const int num_points,
//     const float clip_coe,
//     const float2* __restrict__ means2d,
//     const float3* __restrict__ L_elements,
//     const float* __restrict__ opacities,//
//     const dim3 img_size,
//     const dim3 tile_bounds,
//     const float clip_thresh,
//     // const float opacity_thresh,//
//     float2* __restrict__ xys,
//     float* __restrict__ depths,
//     int* __restrict__ radii,
//     float3* __restrict__ conics,
//     int32_t* __restrict__ num_tiles_hit,
//     bool isprint
// ) {
//     unsigned idx = cg::this_grid().thread_rank(); // idx of thread within grid
//     if (idx >= num_points) {//index超过2D gaussian总数的线程直接返回, 防止数组越界访问
//         return;
//     }
//     radii[idx] = 0;
//     num_tiles_hit[idx] = 0;
//      // 先给一个固定的depth，为了后面的函数调用方便
//      depths[idx] = 0.0f;
//
//     float2 center = {means2d[idx].x ,means2d[idx].y };
//     // Assuming L is packed row-wise in a 1D array: [l11, l21, l22]
//     float l11 = L_elements[idx].x; // scale_x
//     float l21 = L_elements[idx].y; // covariance_xy
//     float l22 = L_elements[idx].z; // scale_y
//
//     float3 cov2d = make_float3(l11, l21, l22); // x的方差，cov(x,y),y的方差
//     // printf("cov2d %d, %.2f %.2f %.2f\n", idx, cov2d.x, cov2d.y, cov2d.z);
//
//     //计算协方差的逆 和 椭圆半径
//     float3 conic;
//     float radius;
//     const float opacity = opacities[idx];
//     bool ok = compute_cov2d_adaptive_radius_bounds(cov2d,opacity, conic, radius, clip_thresh);
//
//     if (!ok) //cov2d不可逆
//         return; // zero determinant
//     if(isprint)
//         printf("gs idx %d conic %.2f %.2f %.2f\n", idx, conic.x, conic.y, conic.z);
//     // printf("conic %d %.2f %.2f %.2f\n", idx, conic.x, conic.y, conic.z);
//     conics[idx] = conic;
//     xys[idx] = center;
//     radii[idx] = (int)radius;
//     uint2 tile_min, tile_max;
//     // 当前gs落在那几个tile上
//     get_tile_bbox(center, radius, tile_bounds, tile_min, tile_max);
//     int32_t tile_area = (tile_max.x - tile_min.x) * (tile_max.y - tile_min.y);//覆盖了多少个tile
//     if (tile_area <= 0) {
//         if (isprint)
//             printf("%d point bbox outside of bounds\n", idx);
//         return;
//     }
//     num_tiles_hit[idx] = tile_area;
//
//
// }
// //axis_aligned_bounds
// __global__ void project_gaussians_2d_covariance_aab_forward_kernel(
//     const int num_points,
//     const float clip_coe,
//     const float2* __restrict__ means2d,
//     const float3* __restrict__ L_elements,
//     const float* __restrict__ opacities,
//     const dim3 img_size,
//     const dim3 tile_bounds,
//     const float clip_thresh,
//     // const float opacity_thresh,
//     float2* __restrict__ xys,
//     float* __restrict__ depths,
//     float2* __restrict__ radii,
//     float3* __restrict__ conics,
//     int32_t* __restrict__ num_tiles_hit,
//     bool isprint
// ) {
//     unsigned idx = cg::this_grid().thread_rank(); // idx of thread within grid
//     if (idx >= num_points) {//index超过2D gaussian总数的线程直接返回, 防止数组越界访问
//         return;
//     }
//     radii[idx].x = 0.0;
//     radii[idx].y= 0.0;
//     num_tiles_hit[idx] = 0;
//      // 先给一个固定的depth，为了后面的函数调用方便
//      depths[idx] = 0.0f;
//
//     float2 center = {means2d[idx].x ,means2d[idx].y };
//     // Assuming L is packed row-wise in a 1D array: [l11, l21, l22]
//     float l11 = L_elements[idx].x; // scale_x
//     float l21 = L_elements[idx].y; // covariance_xy
//     float l22 = L_elements[idx].z; // scale_y
//
//     float3 cov2d = make_float3(l11, l21, l22); // x的方差，cov(x,y),y的方差
//     // printf("cov2d %d, %.2f %.2f %.2f\n", idx, cov2d.x, cov2d.y, cov2d.z);
//
//     //计算协方差的逆 和 椭圆半径
//     float3 conic;
//     float2 radius;
//     const float opacity = opacities[idx];
//     bool ok = compute_cov2d_axis_aligned_bounds(cov2d,opacity, conic, radius, clip_thresh);
//
//     if (!ok) //cov2d不可逆
//         return; // zero determinant
//     if(isprint)
//         printf("gs idx %d conic %.2f %.2f %.2f\n", idx, conic.x, conic.y, conic.z);
//     // printf("conic %d %.2f %.2f %.2f\n", idx, conic.x, conic.y, conic.z);
//     conics[idx] = conic;
//     xys[idx] = center;
//     radii[idx] = radius;
//     uint2 tile_min, tile_max;
//     // 当前gs落在那几个tile上
//     get_tile_bbox_xy(center, radius, tile_bounds, tile_min, tile_max);
//     int32_t tile_area = (tile_max.x - tile_min.x) * (tile_max.y - tile_min.y);//覆盖了多少个tile
//     if (tile_area <= 0) {
//         if (isprint)
//             printf("%d point bbox outside of bounds\n", idx);
//         return;
//     }
//     num_tiles_hit[idx] = tile_area;
//
//
// }

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
// ) {
//     unsigned idx = cg::this_grid().thread_rank(); // idx of thread within grid
//     if (idx >= num_points) {//index超过2D gaussian总数的线程直接返回, 防止数组越界访问
//         return;
//     }
//     radii[idx] = 0;
//     num_tiles_hit[idx] = 0;

//     float2 center = {means2d[idx].x ,means2d[idx].y };
//     // Assuming L is packed row-wise in a 1D array: [l11, l21, l22]
//     float l11 = L_elements[idx].x; // scale_x
//     float l21 = L_elements[idx].y; // covariance_xy
//     float l22 = L_elements[idx].z; // scale_y

//     // Construct the 2x2 covariance matrix from L
//     // float2x2 Cov2D = make_float2x2(l11*l11, l11*l21,
//                                 //    l11*l21, l21*l21 + l22*l22);
//     // float3 cov2d = make_float3(l11*l11, l21, l22*l22);
//     float3 cov2d = make_float3(l11, l21, l22); // x的方差，cov(x,y),y的方差
//     // printf("cov2d %d, %.2f %.2f %.2f\n", idx, cov2d.x, cov2d.y, cov2d.z);
//     float3 conic;
//     float radius;
//     bool ok = compute_cov2d_bounds(cov2d, conic, radius);
    
//     if (!ok) //cov2d不可逆
//         return; // zero determinant
//     if(isprint)
//         printf("gs idx %d conic %.2f %.2f %.2f\n", idx, conic.x, conic.y, conic.z);
//     // printf("conic %d %.2f %.2f %.2f\n", idx, conic.x, conic.y, conic.z);
//     conics[idx] = conic;
//     xys[idx] = center;
//     radii[idx] = (int)radius;
//     uint2 tile_min, tile_max;
//     // 当前gs落在那几个tile上
//     get_tile_bbox(center, radius, tile_bounds, tile_min, tile_max);
//     int32_t tile_area = (tile_max.x - tile_min.x) * (tile_max.y - tile_min.y);//覆盖了多少个tile
//     if (tile_area <= 0) {
//         if (isprint)
//             printf("%d point bbox outside of bounds\n", idx);
//         return;
//     }
//     num_tiles_hit[idx] = tile_area;
//     // 先给一个固定的depth，为了后面的函数调用方便
//     depths[idx] = 0.0f;

// }