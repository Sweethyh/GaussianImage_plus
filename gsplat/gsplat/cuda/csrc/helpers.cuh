#include "config.h"
#include <cuda_runtime.h>
#include "third_party/glm/glm/glm.hpp"
#include "third_party/glm/glm/gtc/type_ptr.hpp"
#include <iostream>
#include <math.h>
//
inline __device__ float ndc2pix(const float x, const float W, const float cx) {
    return 0.5f * W * x + cx - 0.5f;
}
// 计算向量的单位向量
inline __device__ float2 normalize(const float2 &v) {
    float len = sqrtf(v.x * v.x + v.y * v.y);
    return {v.x / len, v.y / len};
}
inline __device__ void get_bbox(
    const float2 center,
    const float2 dims,
    const dim3 img_size,
    uint2 &bb_min,
    uint2 &bb_max
) {
    // get bounding box with center and dims, within bounds
    // bounding box coords returned in tile coords, inclusive min, exclusive max
    // clamp between 0 and tile bounds
    bb_min.x = min(max(0, (int)(center.x - dims.x)), img_size.x);
    bb_max.x = min(max(0, (int)(center.x + dims.x + 1)), img_size.x);
    bb_min.y = min(max(0, (int)(center.y - dims.y)), img_size.y);
    bb_max.y = min(max(0, (int)(center.y + dims.y + 1)), img_size.y);
}

inline __device__ void get_tile_bbox(
    const float2 pix_center,// 椭圆中心
    const float pix_radius,//椭圆半径
    const dim3 tile_bounds,
    uint2 &tile_min,//初始化为0
    uint2 &tile_max
) {
    // gets gaussian dimensions in tile space, i.e. the span of a gaussian in
    // tile_grid (image divided into tiles)
    float2 tile_center = { // 椭圆坐标（全局）映射到 tile 中的坐标 （可以把每个tile看成一个单位像素）

        pix_center.x / (float)BLOCK_X, pix_center.y / (float)BLOCK_Y
    };
  
    float2 tile_radius = {
        pix_radius / (float)BLOCK_X, pix_radius / (float)BLOCK_Y
    };
    get_bbox(tile_center, tile_radius, tile_bounds, tile_min, tile_max);
}
inline __device__ void get_tile_bbox_xy(
    const float2 pix_center,// 椭圆中心
    const float2 pix_radius,//椭圆半径 xy 2个轴不一样
    const dim3 tile_bounds,
    uint2 &tile_min,//初始化为0
    uint2 &tile_max
) {
    // gets gaussian dimensions in tile space, i.e. the span of a gaussian in
    // tile_grid (image divided into tiles)
    float2 tile_center = { // 椭圆坐标（全局）映射到 tile 中的坐标 （可以把每个tile看成一个单位像素）

        pix_center.x / (float)BLOCK_X, pix_center.y / (float)BLOCK_Y
    };
    float2 tile_radius = {
        pix_radius.x / (float)BLOCK_X, pix_radius.y / (float)BLOCK_Y
    };
    get_bbox(tile_center, tile_radius, tile_bounds, tile_min, tile_max);
}

// 以oob bounding 的外接矩形
inline __device__ void get_tile_obb_bbox(
    const float2 pix_min_bound,//椭圆半径 xy 2个轴不一样
    const float2 pix_max_bound,//椭圆半径 xy 2个轴不一样
    const dim3 tile_bounds,
    uint2 &bb_min,//初始化为0
    uint2 &bb_max
) {
    bb_min.x = min(max((int)(pix_min_bound.x/ (float)BLOCK_X) ,0), tile_bounds.x);
    bb_min.y =  min(max((int)(pix_min_bound.y/ (float)BLOCK_Y) ,0), tile_bounds.y);
    bb_max.x =  min(max((int)(pix_max_bound.x/ (float)BLOCK_X+1) ,0), tile_bounds.x);
    bb_max.y =  min(max((int)(pix_max_bound.y/ (float)BLOCK_Y+1) ,0), tile_bounds.y);
}
// CUDA内核函数，计算一个box和一个16x16的tile在一个轴上的投影范围 
inline __device__ bool 
compute_projection_bounds_overlap(float2* vertices, float2* tile_vertices,  float2& axis,bool isprint) { 
    float magnitude = sqrt(axis.x * axis.x + axis.y * axis.y);

    // 如果 magnitude 不为零，则归一化 axis
    if (magnitude != 0.0f) {
        float2 unit_axis = make_float2(axis.x / magnitude, axis.y / magnitude);
        axis = unit_axis;
    } else {
        // 处理 magnitude 为零的情况，例如抛出错误或设置默认值
        // 这里假设设置默认值为 (1.0f, 0.0f)
        axis = make_float2(1.0f, 0.0f);
    }
    float min_proj,  max_proj;
    min_proj = max_proj = vertices[0].x * axis.x + vertices[0].y * axis.y; // axis是单位轴
    for (int i = 1; i < 4; ++i) { 
        float proj = vertices[i].x * axis.x + vertices[i].y * axis.y; 
        if (proj < min_proj) min_proj = proj; 
        if (proj > max_proj) max_proj = proj; 
    }
    float tile_min_proj,  tile_max_proj;
    tile_min_proj = tile_max_proj = tile_vertices[0].x * axis.x + tile_vertices[0].y * axis.y; // axis是单位轴
    for (int i = 1; i < 4; ++i) { 
        float proj = tile_vertices[i].x * axis.x + tile_vertices[i].y * axis.y; 
        if (proj < tile_min_proj) tile_min_proj = proj; 
        if (proj > tile_max_proj) tile_max_proj = proj; 
    }
    if (isprint) { 
        printf("min_proj: %f, max_proj: %f, tile_min_proj: %f, tile_max_proj: %f, "
        "vertices: (%f, %f), (%f, %f), (%f, %f), (%f, %f), "
        "tile_vertices: (%f, %f), (%f, %f), (%f, %f), (%f, %f)\n",
        min_proj, max_proj, tile_min_proj, tile_max_proj,
        vertices[0].x, vertices[0].y, vertices[1].x, vertices[1].y,
        vertices[2].x, vertices[2].y, vertices[3].x, vertices[3].y,
        tile_vertices[0].x, tile_vertices[0].y, tile_vertices[1].x, tile_vertices[1].y,
        tile_vertices[2].x, tile_vertices[2].y, tile_vertices[3].x, tile_vertices[3].y);
}
    //     printf("min_proj: %f, max_proj: %f, tile_min_proj: %f, tile_max_proj: %f\n", min_proj, max_proj, tile_min_proj, tile_max_proj); 
    // }
    return tile_min_proj < max_proj && tile_max_proj > min_proj;
}
// inline __device__ void get_tile_obb_box_intersect(
//     const float2* vertices,
//     const float2 pix_center,// 椭圆中心
//     const float2 pix_radius,//椭圆半径 xy 2个轴不一样
//     const dim3 tile_bounds,
//     uint2 &tile_min,//初始化为0
//     uint2 &tile_max
// ) {
//     float2 long_edge={vertices[0].x - vertices[1].x, vertices[0].y - vertices[1].y};
//     float2& short_edge = {vertices[0].x - vertices[2].x, vertices[0].y - vertices[2].y}; 
//     float2 axes[2] = { 
//     // {1.0f, 0.0f}, // x轴 
//     // {0.0f, 1.0f}, // y轴 
//     normalize(long_edge),
//     normalize(short_edge)
//     };
//     bool overlap = true; 
//     for (int i = 0; i < 2; ++i) { 
//         float min_proj_obb, max_proj_obb; 
//         float min_proj_tile,min_proj_tile;
//     float2 tile_center = { // 椭圆坐标（全局）映射到 tile 中的坐标 （可以把每个tile看成一个单位像素）

//         pix_center.x / (float)BLOCK_X, pix_center.y / (float)BLOCK_Y
//     };
//     float2 tile_radius = {
//         pix_radius.x / (float)BLOCK_X, pix_radius.y / (float)BLOCK_Y
//     };
//     get_bbox(tile_center, tile_radius, tile_bounds, tile_min, tile_max);
// }
// inline __device__ bool
// compute_cov2d_bounds(const float3 cov2d, float3 &conic, float &radius) {
//     // find eigenvalues of 2d covariance matrix
//     // expects upper triangular values of cov matrix as float3
//     // then compute the radius and conic dimensions
//     // the conic is the inverse cov2d matrix, represented here with upper
//     // triangular values.
//     float det = cov2d.x * cov2d.z - cov2d.y * cov2d.y; //协方差行列式
//     if (det == 0.f)//when 协方差矩阵不可逆 不是 半正定
//     // if (det == 0.f)//when 协方差矩阵不可逆 xinjie
//         return false;
//     float inv_det = 1.f / det;

//     // inverse of 2x2 cov2d matrix 伴随矩阵求逆
//     conic.x = cov2d.z * inv_det;
//     conic.y = -cov2d.y * inv_det;
//     conic.z = cov2d.x * inv_det;

//     float b = 0.5f * (cov2d.x + cov2d.z);
//     float v1 = b + sqrt(max(0.1f, b * b - det)); //特征值
//     float v2 = b - sqrt(max(0.1f, b * b - det));//特征值
//     // take 3 sigma of covariance    // 3.f * sqrt(max(v1, v2)) 是3倍标准差，通常用于表示高斯分布的覆盖范围。
//     radius = ceil(3.f * sqrt(max(v1, v2))); //sqrt(max(v1, v2)) 是较大特征值的平方根，表示主轴的标准差
//     return true;
// }
inline __device__ bool
compute_cov2d_bounds(const float3 cov2d, float3 &conic, float2 &radius, const float clip_coe) {
    // find eigenvalues of 2d covariance matrix
    // expects upper triangular values of cov matrix as float3
    // then compute the radius and conic dimensions
    // the conic is the inverse cov2d matrix, represented here with upper
    // triangular values.
    float det = cov2d.x * cov2d.z - cov2d.y * cov2d.y; //协方差行列式
    if (det == 0.f)//when 协方差矩阵不可逆 不是 半正定
    // if (det == 0.f)//when 协方差矩阵不可逆 xinjie
        return false;
    float inv_det = 1.f / det;

    // inverse of 2x2 cov2d matrix 伴随矩阵求逆
    conic.x = cov2d.z * inv_det;
    conic.y = -cov2d.y * inv_det;
    conic.z = cov2d.x * inv_det;

    float b = 0.5f * (cov2d.x + cov2d.z);
    float v1 = b + sqrt(max(0.1f, b * b - det)); //特征值
    float v2 = b - sqrt(max(0.1f, b * b - det));//特征值
    // coeff=3.f
    // take 3 sigma of covariance    // 3.f * sqrt(max(v1, v2)) 是3倍标准差，通常用于表示高斯分布的覆盖范围。
    radius.x = ceil(clip_coe * sqrt(max(v1, v2))); //sqrt(max(v1, v2)) 是较大特征值的平方根，表示主轴的标准差
    radius.y = ceil(clip_coe * sqrt(min(v1, v2))); //sqrt(max(v1, v2)) 是较大特征值的平方根，表示主轴的标准差
    
    return true;
}
inline __device__ bool
compute_cov2d_obb_bounds( float2 &vertices1,float2 &vertices2,float2 &vertices3,float2 &vertices4,
    const float2 xy, const float3 cov2d, 
    float3 &conic, float2 &radius, const float clip_coe) {
    // find eigenvalues of 2d covariance matrix
    // expects upper triangular values of cov matrix as float3
    // then compute the radius and conic dimensions
    // the conic is the inverse cov2d matrix, represented here with upper
    // triangular values.
    float det = cov2d.x * cov2d.z - cov2d.y * cov2d.y; //协方差行列式
    vertices1={-1.0,-1.0};
    vertices2={-1.0,-1.0};
    vertices3={-1.0,-1.0};
    vertices4={-1.0,-1.0};
    if (det == 0.f)//when 协方差矩阵不可逆 不是 半正定
    // if (det == 0.f)//when 协方差矩阵不可逆 xinjie
        return false;
    float inv_det = 1.f / det;

    // inverse of 2x2 cov2d matrix 伴随矩阵求逆
    conic.x = cov2d.z * inv_det;
    conic.y = -cov2d.y * inv_det;
    conic.z = cov2d.x * inv_det;

    float b = 0.5f * (cov2d.x + cov2d.z);
    float v1 = b + sqrt(max(0.1f, b * b - det)); //特征值
    float v2 = b - sqrt(max(0.1f, b * b - det));//特征值
    // 特征向量计算旋转角度
    // float2 eigenvector1 ,eigenvector2;
    // if (cov2d.y != 0) { 
    //     eigenvector1 = make_float2(v1 - cov2d.z, cov2d.y); 
    //     eigenvector2 = make_float2(v2 - cov2d.z, cov2d.y); 
    // } 
    // else { 
    //     eigenvector1 = make_float2(1.0f, 0.0f); 
    //     eigenvector2 = make_float2(0.0f, 1.0f); 
    // }
    float2 eigenvector1;
    if (cov2d.y != 0) { 
        eigenvector1 = make_float2(v1 - cov2d.z, cov2d.y); 
        // eigenvector2 = make_float2(v2 - cov2d.z, cov2d.y); 
    } 
    else { 
        eigenvector1 = make_float2(1.0f, 0.0f); 
        // eigenvector2 = make_float2(0.0f, 1.0f); 
    }
    float theta=atan2f(eigenvector1.y, eigenvector1.x);
    float cos_theta = cosf(theta); 
    float sin_theta = sinf(theta);
    // coeff=3.f
    // take 3 sigma of covariance    // 3.f * sqrt(max(v1, v2)) 是3倍标准差，通常用于表示高斯分布的覆盖范围。
    radius.x = ceil(clip_coe * sqrt(max(v1, v2))); //sqrt(max(v1, v2)) 是较大特征值的平方根，表示主轴的标准差
    radius.y = ceil(clip_coe * sqrt(min(v1, v2))); 
    if (radius.y<2)
        return false;
    // todo 短轴至少占据几个像素？
    // calculate OBB vertices 
    // float2 vertices[4];
    // 椭圆的4个顶点
    // vertices[0] = {radius.x * cos_theta, radius.x  * sin_theta}; //长轴的2个点
    // vertices[1] = {-radius.x  * cos_theta, radius.x  * sin_theta}; 
    // vertices[2] = { radius.y   * sin_theta, - radius.y   * cos_theta}; //短轴的2个点
    // vertices[3] = {- radius.y   * sin_theta,  radius.y   * cos_theta}; // translate vertices by xy 
    // for (int i = 0; i < 4; ++i) { 
    //     vertices[i].x += xy.x; vertices[i].y += xy.y; 
    // }
    // box的4个顶点 （radius.x，radius.y)
    vertices1 = {radius.x * cos_theta-sin_theta * radius.y +xy.x, radius.x  * sin_theta + cos_theta*radius.y+xy.y}; 
    vertices2 = {-radius.x * cos_theta-sin_theta * radius.y +xy.x, -radius.x  * sin_theta + cos_theta*radius.y+xy.y};
    vertices3 ={radius.x * cos_theta+sin_theta * radius.y+xy.x, radius.x  * sin_theta - cos_theta*radius.y+xy.y}; 
    vertices4 ={-radius.x * cos_theta+sin_theta * radius.y+xy.x, -radius.x  * sin_theta - cos_theta*radius.y+xy.y}; 
     // determine min and max bounds min_bound = vertices[0]; max_bound = vertices[0]; for (int i = 1; i < 4; ++i) { if (vertices[i].x < min_bound.x) min_bound.x = vertices[i].x; if (vertices[i].x > max_bound.x) max_bound.x = vertices[i].x; if (vertices[i].y < min_bound.y) min_bound.y = vertices[i].y; if (vertices[i].y > max_bound.y) max_bound.y = vertices[i].y; }
    // determine min and max bounds 
    // min_bound = vertices1; 
    // max_bound = vertices1; 
    // // for (int i = 1; i < 4; ++i) { 
    // if (vertices2.x < min_bound.x) min_bound.x = vertices2.x; 
    // if (vertices2.x > max_bound.x) max_bound.x = vertices2.x; 
    // if (vertices2.y < min_bound.y) min_bound.y = vertices2.y; 
    // if (vertices2.y > max_bound.y) max_bound.y = vertices2.y; 
    // // }
    // if (vertices3.x < min_bound.x) min_bound.x = vertices3.x; 
    // if (vertices3.x > max_bound.x) max_bound.x = vertices3.x; 
    // if (vertices3.y < min_bound.y) min_bound.y = vertices3.y; 
    // if (vertices3.y > max_bound.y) max_bound.y = vertices3.y; 

    // if (vertices4.x < min_bound.x) min_bound.x = vertices4.x; 
    // if (vertices4.x > max_bound.x) max_bound.x = vertices4.x; 
    // if (vertices4.y < min_bound.y) min_bound.y = vertices4.y; 
    // if (vertices4.y > max_bound.y) max_bound.y = vertices4.y; 
    return true;
}
inline __device__ bool
compute_cov2d_bounds_xy(const float3 cov2d, float3 &conic, float2 &radius) {
    // find eigenvalues of 2d covariance matrix
    // expects upper triangular values of cov matrix as float3
    // then compute the radius and conic dimensions
    // the conic is the inverse cov2d matrix, represented here with upper
    // triangular values.
    float det = cov2d.x * cov2d.z - cov2d.y * cov2d.y; //协方差行列式
    if (det == 0.f)//when 协方差矩阵不可逆
        return false;
    // if (cov2d.x<0 || cov2d.z<0)
    //     return false;
    float inv_det = 1.f / det;

    // inverse of 2x2 cov2d matrix 伴随矩阵求逆
    conic.x = cov2d.z * inv_det;
    conic.y = -cov2d.y * inv_det;
    conic.z = cov2d.x * inv_det;

    float b = 0.5f * (cov2d.x + cov2d.z);
    float v1 = b + sqrt(max(0.1f, b * b - det)); //特征值
    float v2 = b - sqrt(max(0.1f, b * b - det));//特征值
    // take 3 sigma of covariance    // 3.f * sqrt(max(v1, v2)) 是3倍标准差，通常用于表示高斯分布的覆盖范围。
    radius.x = (cov2d.x >= cov2d.z) ? ceil(3.f * sqrt(v1)) : ceil(3.f * sqrt(v2)); // sqrt(max(v1, v2)) 是较大特征值的平方根，表示主轴的标准差
    radius.y = (cov2d.x >= cov2d.z) ? ceil(3.f * sqrt(v2)) : ceil(3.f * sqrt(v1));
    return true;
}
inline __device__ bool
compute_cov2d_axis_aligned_bounds(const float3 cov2d,const float opacity, float3 &conic, float2 &radius,const float low_opacity) {
    // find eigenvalues of 2d covariance matrix
    // expects upper triangular values of cov matrix as float3
    // then compute the radius and conic dimensions
    // the conic is the inverse cov2d matrix, represented here with upper
    // triangular values.
    float det = cov2d.x * cov2d.z - cov2d.y * cov2d.y; //协方差行列式
    if (det == 0.f)
        return false;

    float inv_det = 1.f / det;

    // inverse of 2x2 cov2d matrix 伴随矩阵求逆
    conic.x = cov2d.z * inv_det;
    conic.y = -cov2d.y * inv_det;
    conic.z = cov2d.x * inv_det;
   
    float b = 0.5f * (cov2d.x + cov2d.z);
    float v1 = b + sqrt(max(0.1f, b * b - det)); //特征值
    float v2 = b - sqrt(max(0.1f, b * b - det));//特征值
    // take 3 sigma of covariance    // 3.f * sqrt(max(v1, v2)) 是3倍标准差，通常用于表示高斯分布的覆盖范围。
    float radius_o = ceil(3.f * sqrt(max(v1, v2))); //sqrt(max(v1, v2)) 是较大特征值的平方根，表示主轴的标准差
    float coe=2*log(opacity/low_opacity);
    radius.x=cov2d.x>0 ? min(sqrt(coe*cov2d.x),radius_o): radius_o;
    radius.y=cov2d.z>0 ? min(sqrt(coe*cov2d.z),radius_o):radius_o;
    // radius.x=min(sqrt(coe*conic.x),radius_o);
    // radius.y=min(sqrt(coe*conic.z),radius_o);
    return true;
}
inline __device__ bool
compute_cov2d_adaptive_radius_bounds(const float3 cov2d,const float opacity, float3 &conic, float &radius,const float low_opacity) {
    // find eigenvalues of 2d covariance matrix
    // expects upper triangular values of cov matrix as float3
    // then compute the radius and conic dimensions
    // the conic is the inverse cov2d matrix, represented here with upper
    // triangular values.
    float det = cov2d.x * cov2d.z - cov2d.y * cov2d.y; //协方差行列式
    if (det == 0.f)
        return false;
    float inv_det = 1.f / det;

    // inverse of 2x2 cov2d matrix 伴随矩阵求逆
    conic.x = cov2d.z * inv_det;
    conic.y = -cov2d.y * inv_det;
    conic.z = cov2d.x * inv_det;
   
    float b = 0.5f * (cov2d.x + cov2d.z);
    float v1 = b + sqrt(max(0.1f, b * b - det)); //特征值
    float v2 = b - sqrt(max(0.1f, b * b - det));//特征值
    // take 3 sigma of covariance    // 3.f * sqrt(max(v1, v2)) 是3倍标准差，通常用于表示高斯分布的覆盖范围。
    float radius_o = ceil(3.f * sqrt(max(v1, v2))); //sqrt(max(v1, v2)) 是较大特征值的平方根，表示主轴的标准差
    float radius_ad=sqrt(2*max(v1, v2)*log(opacity/low_opacity));
    radius=min(radius_ad,radius_o);
   
    return true;
}
// compute vjp from df/d_conic to df/c_cov2d
inline __device__ void cov2d_to_conic_vjp(
    const float3 &conic, const float3 &v_conic, float3 &v_cov2d
) {
    // conic = inverse cov2d
    // df/d_cov2d = -conic * df/d_conic * conic
    glm::mat2 X = glm::mat2(conic.x, conic.y, conic.y, conic.z);
    glm::mat2 G = glm::mat2(v_conic.x, v_conic.y, v_conic.y, v_conic.z);
    glm::mat2 v_Sigma = -X * G * X;
    v_cov2d.x = v_Sigma[0][0];
    v_cov2d.y = v_Sigma[1][0] + v_Sigma[0][1];
    v_cov2d.z = v_Sigma[1][1];
}

// helper for applying R * p + T, expect mat to be ROW MAJOR
inline __device__ float3 transform_4x3(const float *mat, const float3 p) {
    float3 out = {
        mat[0] * p.x + mat[1] * p.y + mat[2] * p.z + mat[3],
        mat[4] * p.x + mat[5] * p.y + mat[6] * p.z + mat[7],
        mat[8] * p.x + mat[9] * p.y + mat[10] * p.z + mat[11],
    };
    return out;
}

// helper to apply 4x4 transform to 3d vector, return homo coords
// expects mat to be ROW MAJOR
inline __device__ float4 transform_4x4(const float *mat, const float3 p) {
    float4 out = {
        mat[0] * p.x + mat[1] * p.y + mat[2] * p.z + mat[3],
        mat[4] * p.x + mat[5] * p.y + mat[6] * p.z + mat[7],
        mat[8] * p.x + mat[9] * p.y + mat[10] * p.z + mat[11],
        mat[12] * p.x + mat[13] * p.y + mat[14] * p.z + mat[15],
    };
    return out;
}

inline __device__ float2 project_pix(
    const float *mat, const float3 p, const dim3 img_size, const float2 pp
) {
    // ROW MAJOR mat
    float4 p_hom = transform_4x4(mat, p);
    float rw = 1.f / (p_hom.w + 1e-6f);
    float3 p_proj = {p_hom.x * rw, p_hom.y * rw, p_hom.z * rw};
    return {
        ndc2pix(p_proj.x, img_size.x, pp.x), ndc2pix(p_proj.y, img_size.y, pp.y)
    };
}

// given v_xy_pix, get v_xyz
inline __device__ float3 project_pix_vjp(
    const float *mat, const float3 p, const dim3 img_size, const float2 v_xy
) {
    // ROW MAJOR mat
    float4 p_hom = transform_4x4(mat, p);
    float rw = 1.f / (p_hom.w + 1e-6f);

    float3 v_ndc = {0.5f * img_size.x * v_xy.x, 0.5f * img_size.y * v_xy.y};
    float4 v_proj = {
        v_ndc.x * rw, v_ndc.y * rw, 0., -(v_ndc.x + v_ndc.y) * rw * rw
    };
    // df / d_world = df / d_cam * d_cam / d_world
    // = v_proj * P[:3, :3]
    return {
        mat[0] * v_proj.x + mat[4] * v_proj.y + mat[8] * v_proj.z,
        mat[1] * v_proj.x + mat[5] * v_proj.y + mat[9] * v_proj.z,
        mat[2] * v_proj.x + mat[6] * v_proj.y + mat[10] * v_proj.z
    };
}

inline __device__ glm::mat3 quat_to_rotmat(const float4 quat) {
    // quat to rotation matrix
    float s = rsqrtf(
        quat.w * quat.w + quat.x * quat.x + quat.y * quat.y + quat.z * quat.z
    );
    float w = quat.x * s;
    float x = quat.y * s;
    float y = quat.z * s;
    float z = quat.w * s;

    // glm matrices are column-major
    return glm::mat3(
        1.f - 2.f * (y * y + z * z),
        2.f * (x * y + w * z),
        2.f * (x * z - w * y),
        2.f * (x * y - w * z),
        1.f - 2.f * (x * x + z * z),
        2.f * (y * z + w * x),
        2.f * (x * z + w * y),
        2.f * (y * z - w * x),
        1.f - 2.f * (x * x + y * y)
    );
}

// inline __device__ glm::mat3 rotor_to_rotmat(const float4 rot) {
//     // quat to rotation matrix
//     float s = rsqrtf(
//         rot.x * rot.x + rot.y * rot.y + rot.z * rot.z + rot.w * rot.w
//     );
//     float x = rot.x * s;
//     float y = rot.y * s;
//     float z = rot.z * s;
//     float w = rot.w * s;

//     // glm matrices are column-major
//     return glm::mat3(
//         x * x - y * y - z * z + w * w,
//         -2.f * (x * y + w * z),
//         2.f * (y * w - x * z),
//         2.f * (x * y - w * z),
//         x * x - y * y + z * z - w * w,
//         -2.f * (y * z + w * x),
//         2.f * (y * w + x * z),
//         2.f * (x * w - y * z),
//         x * x + y * y - z * z - w * w
//     );
// }



inline __device__ float4
quat_to_rotmat_vjp(const float4 quat, const glm::mat3 v_R) {
    float s = rsqrtf(
        quat.w * quat.w + quat.x * quat.x + quat.y * quat.y + quat.z * quat.z
    );
    float w = quat.x * s;
    float x = quat.y * s;
    float y = quat.z * s;
    float z = quat.w * s;

    float4 v_quat;
    // v_R is COLUMN MAJOR
    // w element stored in x field
    v_quat.x =
        2.f * (
                  // v_quat.w = 2.f * (
                  x * (v_R[1][2] - v_R[2][1]) + y * (v_R[2][0] - v_R[0][2]) +
                  z * (v_R[0][1] - v_R[1][0])
              );
    // x element in y field
    v_quat.y =
        2.f *
        (
            // v_quat.x = 2.f * (
            -2.f * x * (v_R[1][1] + v_R[2][2]) + y * (v_R[0][1] + v_R[1][0]) +
            z * (v_R[0][2] + v_R[2][0]) + w * (v_R[1][2] - v_R[2][1])
        );
    // y element in z field
    v_quat.z =
        2.f *
        (
            // v_quat.y = 2.f * (
            x * (v_R[0][1] + v_R[1][0]) - 2.f * y * (v_R[0][0] + v_R[2][2]) +
            z * (v_R[1][2] + v_R[2][1]) + w * (v_R[2][0] - v_R[0][2])
        );
    // z element in w field
    v_quat.w =
        2.f *
        (
            // v_quat.z = 2.f * (
            x * (v_R[0][2] + v_R[2][0]) + y * (v_R[1][2] + v_R[2][1]) -
            2.f * z * (v_R[0][0] + v_R[1][1]) + w * (v_R[0][1] - v_R[1][0])
        );
    return v_quat;
}

inline __device__ glm::mat3
scale_to_mat(const float3 scale, const float glob_scale) {
    glm::mat3 S = glm::mat3(1.f);
    S[0][0] = glob_scale * scale.x;
    S[1][1] = glob_scale * scale.y;
    S[2][2] = glob_scale * scale.z;
    return S;
}

// inline __device__ glm::mat3
// inverse_scale_to_mat(const float3 scale, const float glob_scale) {
//     glm::mat3 S = glm::mat3(1.f);
//     S[0][0] = 1 / (glob_scale * scale.x);
//     S[1][1] = 1 / (glob_scale * scale.y);
//     S[2][2] = 1 / (glob_scale * scale.z);
//     return S;
// }

inline __device__ glm::mat3
triangular_mat(const float3 diag_elements, const float3 non_diag_elements) {
    glm::mat3 L = glm::mat3(1.f);
    L[0][0] = diag_elements.x;
    L[1][1] = diag_elements.y;
    L[2][2] = diag_elements.z;
    L[1][0] = non_diag_elements.x;
    L[2][0] = non_diag_elements.y;
    L[2][1] = non_diag_elements.z;
    return L;
}


inline __device__ glm::mat2
scale_to_mat2d(const float2 scale) {
    glm::mat2 S = glm::mat2(1.f);
    S[0][0] = scale.x;
    S[1][1] = scale.y;
    return S;
}

inline __device__ glm::mat2 rotmat2d(const float rot) {
    // quat to rotation matrix
    float cosr = cos(rot);
    float sinr = sin(rot);

    glm::mat2 R = glm::mat2(cosr);
    R[0][1] = -sinr;
    R[1][0] = sinr;

    // glm matrices are column-major
    return R;
}

inline __device__ glm::mat2 rotmat2d_gradient(const float rot) {
    // quat to rotation matrix
    float cosr = cos(rot);
    float sinr = sin(rot);

    glm::mat2 R = glm::mat2(-sinr);
    R[0][1] = -cosr;
    R[1][0] = cosr;

    // glm matrices are column-major
    return R;
}

// device helper for culling near points
inline __device__ bool clip_near_plane(
    const float3 p, const float *viewmat, float3 &p_view, float thresh
) {
    p_view = transform_4x3(viewmat, p);
    if (p_view.z <= thresh) {
        return true;
    }
    return false;
}

inline __device__ int lower_bound_cu(const float *array, int size, float key)
{
    int first = 0, len = size;
    int half, middle;

    while(len > 0){
        half = len >> 1;
        middle = first + half;
        if(array[middle] < key){
            first = middle + 1;
            len = len - half - 1;
        }
        else{
            len = half;
        }
    }
    return first;
}