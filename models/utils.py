import os
import torch.nn.functional as F
from pytorch_msssim import ms_ssim, ssim
import torch
import numpy as np
import math
from errno import EEXIST
from os import makedirs, path
from typing import NamedTuple
# import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import cv2
from PIL import Image
# import os
import wandb
def log_metric_to_wandb(key, _object, step):
    wandb.log({key: _object}, step=step)#, commit=False)
def hwc_to_chw(img):
    """Converts [H,W,C] to [C,H,W] for TensorBoard output.

    Args:
        img (torch.Tensor): [H,W,C] image.

    Returns:
        (torch.Tensor): [C,H,W] image.
    """
    if isinstance(img, np.ndarray):
        img=torch.from_numpy(img)
    return img.permute(2, 0, 1)
    # return img
def log_images_to_wandb(key, image, step):
        # image=hwc_to_chw(image)
    # if not log_metrics_only:
        # wandb.log({key: wandb.Image(np.moveaxis(image, 0, -1))}, step=step, commit=False)
        wandb.log({key: wandb.Image(image)},step=step)#, commit=False)
class LogWriter:
    def __init__(self, file_path, train=True):
        os.makedirs(file_path, exist_ok=True)
        self.file_path = os.path.join(file_path, "train.txt" if train else "test.txt")

    def write(self, text):
        # 打印到控制台
        print(text)
        # 追加到文件
        with open(self.file_path, 'a') as file:
            file.write(text + '\n')
class LogWandb:
    def __init__(self, file_path, train=True):
        os.makedirs(file_path, exist_ok=True)
        # self.file_path = os.path.join(file_path, "train.txt" if train else "test.txt")

    # def write(self, text):
    #     # 打印到控制台
    #     print(text)
    #     # 追加到文件
    #     with open(self.file_path, 'a') as file:
    #         file.write(text + '\n')

def loss_fn(pred, target, loss_type='L2', lambda_value=0.7):
    target = target.detach()
    pred = pred.float()
    target  = target.float()
    if loss_type == 'L2':
        loss = F.mse_loss(pred, target)
    elif loss_type == 'L1':
        loss = F.l1_loss(pred, target)
    elif loss_type == 'SSIM':
        loss = 1 - ssim(pred, target, data_range=1, size_average=True)
    elif loss_type == 'Fusion1':
        loss = lambda_value * F.mse_loss(pred, target) + (1-lambda_value) * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion2':
        loss = lambda_value * F.l1_loss(pred, target) + (1-lambda_value) * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion3':
        loss = lambda_value * F.mse_loss(pred, target) + (1-lambda_value) * F.l1_loss(pred, target)
    elif loss_type == 'Fusion4':
        loss = lambda_value * F.l1_loss(pred, target) + (1-lambda_value) * (1 - ms_ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion_hinerv':
        loss = lambda_value * F.l1_loss(pred, target) + (1-lambda_value)  * (1 - ms_ssim(pred, target, data_range=1, size_average=True, win_size=5))
    return loss

def strip_lowerdiag(L):
    if L.shape[1] == 3:
        uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")
        uncertainty[:, 0] = L[:, 0, 0]
        uncertainty[:, 1] = L[:, 0, 1]
        uncertainty[:, 2] = L[:, 0, 2]
        uncertainty[:, 3] = L[:, 1, 1]
        uncertainty[:, 4] = L[:, 1, 2]
        uncertainty[:, 5] = L[:, 2, 2]

    elif L.shape[1] == 2:
        uncertainty = torch.zeros((L.shape[0], 3), dtype=torch.float, device="cuda")
        uncertainty[:, 0] = L[:, 0, 0]
        uncertainty[:, 1] = L[:, 0, 1]
        uncertainty[:, 2] = L[:, 1, 1]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    '''
        将四元数
    '''
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def build_rotation_2d(r):
    '''
    Build rotation matrix in 2D.
    '''
    R = torch.zeros((r.size(0), 2, 2), device='cuda')
    R[:, 0, 0] = torch.cos(r)[:, 0]
    R[:, 0, 1] = -torch.sin(r)[:, 0]
    R[:, 1, 0] = torch.sin(r)[:, 0]
    R[:, 1, 1] = torch.cos(r)[:, 0]
    return R

def build_scaling_rotation_2d(s, r, device):
    L = torch.zeros((s.shape[0], 2, 2), dtype=torch.float, device='cuda')
    R = build_rotation_2d(r, device)
    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L = R @ L
    return L
    
def build_covariance_from_scaling_rotation_2d(scaling, scaling_modifier, rotation, device):
    '''
    Build covariance metrix from rotation and scale matricies.
    '''
    L = build_scaling_rotation_2d(scaling_modifier * scaling, rotation, device)
    actual_covariance = L @ L.transpose(1, 2)
    return actual_covariance

def build_triangular(r):
    R = torch.zeros((r.size(0), 2, 2), device=r.device)
    R[:, 0, 0] = r[:, 0]
    R[:, 1, 0] = r[:, 1]
    R[:, 1, 1] = r[:, 2]
    return R

#  graphics_utils
class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

# sh utils
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]   


def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                C1 * y * sh[..., 1] +
                C1 * z * sh[..., 2] -
                C1 * x * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                    C2[0] * xy * sh[..., 4] +
                    C2[1] * yz * sh[..., 5] +
                    C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                    C2[3] * xz * sh[..., 7] +
                    C2[4] * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result +
                C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                C3[1] * xy * z * sh[..., 10] +
                C3[2] * y * (4 * zz - xx - yy)* sh[..., 11] +
                C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                C3[5] * z * (xx - yy) * sh[..., 14] +
                C3[6] * x * (xx - 3 * yy) * sh[..., 15])

                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                            C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                            C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                            C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                            C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                            C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                            C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                            C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                            C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result

def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    return sh * C0 + 0.5

# system utils


def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise

def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)

# general utils
def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper
def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)
    

    # visualiz
def create_heatmap(H, W, xys, values, radius=1,image_radius=None):
    """
    创建热图。
    :param H: 图像高度
    :param W: 图像宽度
    :param xys: 点的坐标数组
    :param values: 点的属性值数组
    :param radius: 每个点的影响范围
    :return: 热图
    """
    heatmap = np.zeros((H, W), dtype=np.float32)
    xs=[]
    ys=[]
    v=[]
    visited=[]
    # 创建一个新的图形
    # fig, ax = plt.subplots(figsize=(10, 10))
    for i in range(len(xys)):
        coord = xys[i]
        value = values[i]
        x, y = int(coord[0]), int(coord[1])
        if x>=W or y>=H: continue
        # heatmap[max(0, y - radius):min(H, y + radius + 1), max(0, x - radius):min(W, x + radius + 1)] = value
        if (x,y) in visited: continue
        xs.append(x)
        ys.append(y)
        v.append(value)
        visited.append((x,y))
        heatmap[max(0, y), max(0, x )] +=1
        # circle = plt.Circle((x, y), v, color=colors[i], fill=True)
        # ax.add_patch(circle)
        # heatmap[max(0, y), max(0, x )] +=value
    # 绘制散点图
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=z, cmap='viridis', marker='o')

    # 设置轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Num Tiles Hit')

    # 反转 y 轴，使坐标原点在左上角
    ax.invert_yaxis()

    # 保存图像
    plt.savefig("num_tiles_hit.png")

    # 显示图像
    plt.show()
      # 绘制半径
    for i in range(H):
        for j in range(W):
            if heatmap[i, j] > 0:
                # heatmap[i, j] = 1
                cv2.circle(image_radius, (i,j), int( heatmap[i, j]), (0,0,255), 1)

    return heatmap
def visualize_3d(xys, num_tiles_hit,log_dir):
        """
        可视化三维散点图。
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # def remove_duplicate_points(xys, num_tiles_hit):
        #     """
        #     去除重复的 x, y 坐标，并合并 num_tiles_hit。
        #     """
        #     unique_xys, unique_indices = np.unique(xys, axis=0, return_index=True)
        #     unique_num_tiles_hit = np.zeros(unique_xys.shape[0])

        #     for i, idx in enumerate(unique_indices):
        #         mask = np.all(xys == unique_xys[i], axis=1)
        #         unique_num_tiles_hit[i] = np.sum(num_tiles_hit[mask])

        #     return unique_xys, unique_num_tiles_hit
        # 提取 x, y, z 坐标
        x = xys[:, 0]
        y = xys[:, 1]
        z = num_tiles_hit

        # 绘制散点图
        ax.scatter(x, y, z, c=z, cmap='viridis', marker='o')

        # 设置轴标签
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Num Tiles Hit')
        # 反转 y 轴，使坐标原点在左上角
        # ax.invert_yaxis()

        # 保存图像
        plt.savefig(f"{log_dir}/num_tiles_hit_{len(x)}.png")
        plt.show()
def visual_points_xyz(log_dir,coords,iter,H,W,color=None
                 ):
    # image = np.ones((H, W, 3), dtype=np.uint8) * 255  # 白色背景
    image_radius = np.ones((H, W, 3), dtype=np.uint8) *0
    # image_tiles_hit = np.ones((H, W, 3), dtype=np.uint8) * 255
    invalid_gs=0
    valid_indices = []
    # plt.figure(figsize=(10, 6))
    init_num_points=len(coords)
    for i in range(init_num_points):
        coord = coords[i]
        # cov = covs[i]
        # color = np.clip(colors[i]*255.,0,255).astype(np.uint8)
        
        # 绘制点
        # c=tuple(color.astype(uint8))
        c=tuple([int(x) for x in color[i]]) if color is not None else (255,0,0)
        # c=(255,0,0)
        # 绘制椭圆
        # cv2.ellipse(image, (int(coord[0]), int(coord[1])), (int(width), int(height)), angle, 0, 360, c, 2)
          # 绘制半径
        cv2.circle(image_radius, (int(coord[0]), int(coord[1])), 2, c, -1)
        # cv2.circle( image_tiles_hit, (int(coord[0]), int(coord[1])), int(radius[i]), c, -1)

        # plt.scatter(coord[0], coord[1],  zorder=2)
        # 绘制命中 tile 数量
        # cv2.ellipse(image_tiles_hit, (int(coord[0]), int(coord[1])), (int(num_tile_hit[i]), int(num_tile_hit[i])), angle, 0, 360,c, 0.1)
        # 标注坐标
        # cv2.putText(image, f'({coord[0]:.2f}, {coord[1]:.2f})', (int(coord[0]), int(coord[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
        # # 计算协方差矩阵的特征值和特征向量
        # cov_matrix = np.array([[cov[0], cov[1]], [cov[1], cov[2]]])
        # eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        # angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        # width, height = 2 * np.sqrt(eigenvalues)
        
        # # 绘制椭圆
        # ellipse = Ellipse(xy=coord, width=width, height=height, angle=angle, edgecolor=color, facecolor='none', linewidth=2, zorder=1)
        # plt.gca().add_patch(ellipse)
        
        # # 标注坐标
        # plt.text(coord[0], coord[1], f'({coord[0]:.2f}, {coord[1]:.2f})', fontsize=8, ha='right', va='bottom', color=color)
    grid_size = 16
    x_ticks = np.linspace(0, W, grid_size + 1)
    y_ticks = np.linspace(0, H, grid_size + 1)

    # 绘制水平网格线
    for y in y_ticks:
        # cv2.line(image, (0, int(y)), (W, int(y)), (128, 128, 128), 1)
        cv2.line(image_radius, (0, int(y)), (W, int(y)), (128, 128, 128), 1)
        # cv2.line(image_tiles_hit, (0, int(y)), (W, int(y)), (128, 128, 128), 1)

#      # 绘制散点图
#     plt.figure(figsize=(10, 6))
# #   tter(coord[0], coord[1], color=color, zorder=2)
#     plt.title('Scatter Plot of Points')
#     plt.xlabel('X Coordinate')
#     plt.ylabel('Y Coordinate')
#     plt.axis('equal')
#     plt.grid(True)
#     plt.savefig('scatter_plot.png')
#     plt.show()
    # 绘制
    # 绘制垂直网格线
    for x in x_ticks:
        # cv2.line(image, (int(x), 0), (int(x), H), (128, 128, 128), 1)
        cv2.line(image_radius, (int(x), 0), (int(x), H), (128, 128, 128), 1)
        # cv2.line(image_tiles_hit, (int(x), 0), (int(x), H), (128, 128, 128), 1)
    # plt.figure(figsize=(10, 10))
    # plt.imshow(cv2.cvtColor(image_radius, cv2.COLOR_BGR2RGB))
    # plt.title('coords')
    # plt.axis('off')  # 关闭坐标轴
    # plt.show()
    print(f"invalid_gs with negative eigenvalues:{invalid_gs}")
    cv2.imwrite(f'{log_dir}/coord_gs_points_N{init_num_points}_I{iter}.png', image_radius)
    # cv2.imwrite(f'{log_dir}/radius_image_iter{init_num_points}.png', image_radius)
    # cv2.imwrite(f'{log_dir}/tiles_hit_image_iter{init_num_points}.png', image_tiles_hit)
    # plt.savefig(f"{log_dir}/coord_xy_I{init_num_points}.png")
    # valid_xys =coords[valid_indices]
    # valid_radii = radius[valid_indices]
    # valid_num_tiles_hit = num_tile_hit[valid_indices]
    return cv2.cvtColor(image_radius, cv2.COLOR_BGR2RGB)
def visual_points(log_dir,coords,covs,colors,init_num_points,iter,H,W,
                  radius,num_tile_hit):
    image = np.ones((H, W, 3), dtype=np.uint8) * 0 # 白色背景
    image_radius = np.ones((H, W, 3), dtype=np.uint8) * 0
    image_tiles_hit = np.ones((H, W, 3), dtype=np.uint8) * 0
    invalid_gs=0
    valid_indices = []
    plt.figure(figsize=(10, 6))
        
    for i in range(init_num_points):
        coord = coords[i]
        cov = covs[i]
        color = np.clip(colors[i]*255.,0,255).astype(np.uint8)
        
        # 绘制点
        # c=tuple(color.astype(uint8))
        c=tuple([int(x) for x in color])
        # cv2.circle(image, (int(coord[0]), int(coord[1])), 3, c, -1)
        # plt.scatter(coord[0], coord[1], color=color, zorder=2)
            # 计算协方差矩阵的特征值和特征向量
        cov_matrix = np.array([[cov[0], cov[1]], [cov[1], cov[2]]])
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        # valid_indices.append(i)
        if (eigenvalues<=0).any(): #协方差矩阵的特征值一定非负  特征值的平方根对应于椭圆的半轴长度
            invalid_gs+=1
            continue
        else:
           valid_indices.append(i)
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        width, height = 2 * np.sqrt(eigenvalues)
        
        # 绘制椭圆
        cv2.ellipse(image, (int(coord[0]), int(coord[1])), (int(width), int(height)), angle, 0, 360, c, 2)
          # 绘制半径
        cv2.circle(image_radius, (int(coord[0]), int(coord[1])), int(radius[i]), c, -1)
        cv2.circle( image_tiles_hit, (int(coord[0]), int(coord[1])), int(radius[i]), c, -1)

        # plt.scatter(coord[0], coord[1],  zorder=2)
        # 绘制命中 tile 数量
        # cv2.ellipse(image_tiles_hit, (int(coord[0]), int(coord[1])), (int(num_tile_hit[i]), int(num_tile_hit[i])), angle, 0, 360,c, 0.1)
        # 标注坐标
        # cv2.putText(image, f'({coord[0]:.2f}, {coord[1]:.2f})', (int(coord[0]), int(coord[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
        # # 计算协方差矩阵的特征值和特征向量
        # cov_matrix = np.array([[cov[0], cov[1]], [cov[1], cov[2]]])
        # eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        # angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        # width, height = 2 * np.sqrt(eigenvalues)
        
        # # 绘制椭圆
        # ellipse = Ellipse(xy=coord, width=width, height=height, angle=angle, edgecolor=color, facecolor='none', linewidth=2, zorder=1)
        # plt.gca().add_patch(ellipse)
        
        # # 标注坐标
        # plt.text(coord[0], coord[1], f'({coord[0]:.2f}, {coord[1]:.2f})', fontsize=8, ha='right', va='bottom', color=color)
    grid_size = 16
    x_ticks = np.linspace(0, W, grid_size + 1)
    y_ticks = np.linspace(0, H, grid_size + 1)

    # 绘制水平网格线
    for y in y_ticks:
        cv2.line(image, (0, int(y)), (W, int(y)), (128, 128, 128), 1)
        cv2.line(image_radius, (0, int(y)), (W, int(y)), (128, 128, 128), 1)
        cv2.line(image_tiles_hit, (0, int(y)), (W, int(y)), (128, 128, 128), 1)

#      # 绘制散点图
#     plt.figure(figsize=(10, 6))
# #   tter(coord[0], coord[1], color=color, zorder=2)
#     plt.title('Scatter Plot of Points')
#     plt.xlabel('X Coordinate')
#     plt.ylabel('Y Coordinate')
#     plt.axis('equal')
#     plt.grid(True)
#     plt.savefig('scatter_plot.png')
#     plt.show()
    # 绘制
    # 绘制垂直网格线
    for x in x_ticks:
        cv2.line(image, (int(x), 0), (int(x), H), (128, 128, 128), 1)
        cv2.line(image_radius, (int(x), 0), (int(x), H), (128, 128, 128), 1)
        cv2.line(image_tiles_hit, (int(x), 0), (int(x), H), (128, 128, 128), 1)
    # plt.figure(figsize=(10, 10))
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.title('2d Covariance Ellipses')
    # plt.axis('off')  # 关闭坐标轴
    # plt.show()
    print(f"invalid_gs with negative eigenvalues:{invalid_gs}")
    cv2.imwrite(f'{log_dir}/visual_gs_points_N{init_num_points}_I{iter}.png', image)
    cv2.imwrite(f'{log_dir}/radius_image_N{init_num_points}_I{iter}.png', image_radius)
    cv2.imwrite(f'{log_dir}/tiles_hit_image_N{init_num_points}_I{iter}.png', image_tiles_hit)

    valid_xys =coords[valid_indices]
    valid_radii = radius[valid_indices]
    valid_num_tiles_hit = num_tile_hit[valid_indices]
   
    # # 创建图形
    # fig = plt.figure(figsize=(15, 7))  # 调整图像大小

    # # 创建第一个子图：显示 radii
    # ax1 = fig.add_subplot(121, projection='3d')
    # scatter1 = ax1.scatter(valid_xys[:, 0], valid_xys[:, 1], valid_radii, c=valid_radii, cmap='viridis', s=50, edgecolors='k')

    # # 添加标签
    # for i in range(len(valid_xys)):
    #     coord = valid_xys[i]
    #     radius = valid_radii[i]
    #     ax1.text(coord[0], coord[1], radius, f'{radius:.1f}', color='black', fontsize=10, ha='center', va='bottom')

    # # 设置轴标签和标题
    # ax1.set_xlabel('X Coordinate')
    # ax1.set_ylabel('Y Coordinate')
    # ax1.set_zlabel('Radius')
    # ax1.set_title('3D Map Visualization of Radii')

    # # 添加颜色条
    # cbar1 = fig.colorbar(scatter1, ax=ax1, shrink=0.5, aspect=5)
    # cbar1.set_label('Radius')

    # # 创建第二个子图：显示 num_tiles_hit
    # ax2 = fig.add_subplot(122, projection='3d')
    # scatter2 = ax2.scatter(valid_xys[:, 0], valid_xys[:, 1], valid_num_tiles_hit, c=valid_num_tiles_hit, cmap='plasma', s=50, edgecolors='k')

    # # # 添加标签
    # # for i in range(len(valid_xys)):
    # #     coord = valid_xys[i]
    # #     tiles_hit = valid_num_tiles_hit[i]
    # #     ax2.text(coord[0], coord[1], tiles_hit, str(tiles_hit), color='black', fontsize=10, ha='center', va='bottom')

    # # 设置轴标签和标题
    # ax2.set_xlabel('X Coordinate')
    # ax2.set_ylabel('Y Coordinate')
    # ax2.set_zlabel('Tiles Hit')
    # ax2.set_title('3D Map Visualization of Tiles Hit')

    # # 添加颜色条
    # cbar2 = fig.colorbar(scatter2, ax=ax2, shrink=0.5, aspect=5)
    # cbar2.set_label('Tiles Hit')

    # # 保存图像到本地
    # plt.savefig('3d_map_with_two_plots.png')

    # plt.show()
    # # 创建图形
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))  # 调整图像大小

    # 创建高度图（Radii）
    plt.figure(figsize=(10, 7))
    height_map = np.zeros((H, W), dtype=np.float32)
    height_map_point = np.zeros((H, W), dtype=np.float32)
    for i in range(len(valid_xys)):
        coord = valid_xys[i]
        radius = valid_radii[i]
        x, y = int(coord[0]), int(coord[1])
        height_map[y - radius:y + radius, x - radius:x + radius] = radius
        if x >= 0 and y >= 0 and x < W and y < H:
            height_map_point[y, x] = radius
    image = Image.fromarray(height_map_point.astype(np.uint8))
    image.save(f'{log_dir}/radius_point{init_num_points}.png')
    image = Image.fromarray(height_map.astype(np.uint8))
    image.save(f'{log_dir}/radius_gt{init_num_points}.png')
    plt.figure(figsize=(10, 7))
    plt.imshow(height_map)
    plt.title("the radius of each gs")
    plt.colorbar()
    plt.savefig(f'{log_dir}/radius_{init_num_points}.png')
    plt.figure(figsize=(10, 7))
    plt.imshow(height_map_point)
    plt.title("the radius of each gs")
    plt.colorbar()
    plt.savefig(f'{log_dir}/gs_radius_{init_num_points}.png')
    return height_map,cv2.cvtColor(image_radius, cv2.COLOR_BGR2RGB)
    # # 绘制高度图
    # im1 = ax1.imshow(height_map, cmap='viridis', origin='lower', extent=[0, W, 0, H])
    # ax1.set_title('Map Visualization of Radii')
    # ax1.set_xlabel('X Coordinate')
    # ax1.set_ylabel('Y Coordinate')
    # cbar1 = fig.colorbar(im1, ax=ax1, shrink=0.5, aspect=5)
    # cbar1.set_label('Radius')

    # # 创建热图（Num Tiles Hit）
    # heatmap = np.zeros((H, W), dtype=np.float32)
    # for i in range(len(valid_xys)):
    #     coord = valid_xys[i]
    #     tiles_hit = valid_num_tiles_hit[i]
    #     x, y = int(coord[0]), int(coord[1])
    #     heatmap[y - 1:y + 2, x - 1:x + 2] = tiles_hit  # 假设每个点影响周围 3x3 区域

    # # 绘制热图
    # im2 = ax2.imshow(heatmap, cmap='plasma', origin='lower', extent=[0, W, 0, H])
    # ax2.set_title('Heatmap of Tiles Hit')
    # ax2.set_xlabel('X Coordinate')
    # ax2.set_ylabel('Y Coordinate')
    # cbar2 = fig.colorbar(im2, ax=ax2, shrink=0.5, aspect=5)
    # cbar2.set_label('Tiles Hit')

    # # 保存图像到本地
    # plt.savefig('maps_with_heatmap.png')

    # # 创建白色背景图像
    # background_image = np.ones((H, W, 3), dtype=np.uint8) * 255

    # # 创建高度图（Radii）
    # height_map = create_heatmap(H, W, valid_xys, valid_radii, radius=5,image_radius=image_radius)
    # # 可视化高度图
    # plt.figure(figsize=(10, 6))
    # im = plt.imshow(height_map, cmap='viridis', interpolation='bilinear')
    # plt.title('Height Map (Radii)')
    # plt.colorbar(im, orientation='vertical', label='Radius')
    # plt.axis('off')
    # plt.savefig('height_map_radii.png')
    # plt.show()

    # # # 将高度图转换为热图颜色
    # # height_map_color = cv2.applyColorMap((height_map * 255 / height_map.max()).astype(np.uint8), cv2.COLORMAP_VIRIDIS)

    # # # 叠加高度图到白色背景图像
    # # combined_height_map = cv2.addWeighted(background_image, 0.5, height_map_color, 0.5, 0)

    # # # 保存高度图
    # # cv2.imwrite(f'heatmap_radii.png', combined_height_map)

    # # 创建热图（Num Tiles Hit）
    # heatmap = create_heatmap(H, W, valid_xys, valid_num_tiles_hit, radius=1,image_radius=image_tiles_hit)
    #  # 可视化高度图
    # plt.figure(figsize=(10, 6))
    # im = plt.imshow(height_map, cmap='viridis', interpolation='bilinear')
    # plt.title('Heatmap (Num Tiles Hit)')
    # plt.colorbar(im, orientation='vertical', label='Number of Tiles Hit')
    # plt.axis('off')
    # plt.savefig('height_map_Tiles Hit.png')
    # plt.show()
    
    # # 将热图转换为热图颜色
    # heatmap_color = cv2.applyColorMap((heatmap * 255 / heatmap.max()).astype(np.uint8), cv2.COLORMAP_PLASMA)

    # # 叠加热图到白色背景图像
    # combined_heatmap = cv2.addWeighted(background_image, 0.5, heatmap_color, 0.5, 0)

    # # 保存热图
    # cv2.imwrite(f'heatmap_tiles_hit.png', combined_heatmap)
    # # 显示高度图并添加颜色条
    # fig, ax = plt.subplots(figsize=(10, 6))
    # im = ax.imshow(cv2.cvtColor(combined_height_map, cv2.COLOR_BGR2RGB))
    # ax.set_title('Height Map (Radii)')
    # cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis), ax=ax, orientation='vertical')
    # cbar.set_label('Radius')
    # ax.axis('off')
    # plt.savefig('heatmap_radii_with_colorbar.png')
    # plt.show()

    # # 显示热图并添加颜色条
    # fig, ax = plt.subplots(figsize=(10, 6))
    # im = ax.imshow(cv2.cvtColor(combined_heatmap, cv2.COLOR_BGR2RGB))
    # ax.set_title('Heatmap (Num Tiles Hit)')
    # cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.plasma), ax=ax, orientation='vertical')
    # cbar.set_label('Number of Tiles Hit')
    # ax.axis('off')
    # plt.savefig('heatmap_tiles_hit_with_colorbar.png')
    # plt.show()

    # visualize_3d(coords[valid_indices],valid_num_tiles_hit,log_dir)

def visual_gs_points(log_dir,coords,covs,colors,init_num_points,H,W,
                  radius,num_tile_hit,per_pix_gs_n,iter):
    # image = np.ones((H, W, 3), dtype=np.uint8) * 255  # 白色背景
    # image_radius = np.ones((H, W, 3), dtype=np.uint8) * 0
    # # image_tiles_hit = np.ones((H, W, 3), dtype=np.uint8) * 255
    # invalid_gs=0
    # valid_indices = []
    # plt.figure(figsize=(10, 6))
        
    # for i in range(init_num_points):
    #     coord = coords[i]
    #     cov = covs[i]
    #     color = np.clip(colors[i]*255.,0,255).astype(np.uint8)
        
    #     # 绘制点
    #     # c=tuple(color.astype(uint8))
    #     c=tuple([int(x) for x in color])
    #     # cv2.circle(image, (int(coord[0]), int(coord[1])), 3, c, -1)
    #     # plt.scatter(coord[0], coord[1], color=color, zorder=2)
    #         # 计算协方差矩阵的特征值和特征向量
    #     cov_matrix = np.array([[cov[0], cov[1]], [cov[1], cov[2]]])
    #     eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    #     # valid_indices.append(i)
    #     if (eigenvalues<=0).any(): #协方差矩阵的特征值一定非负  特征值的平方根对应于椭圆的半轴长度
    #         invalid_gs+=1
    #         continue
    #     else:
    #        valid_indices.append(i)
    #     angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    #     width, height = 2 * np.sqrt(eigenvalues)
        
    #     # # 绘制椭圆
    #     # cv2.ellipse(image, (int(coord[0]), int(coord[1])), (int(width), int(height)), angle, 0, 360, c, 2)
    #     #   # 绘制半径
    #     # cv2.circle(image_radius, (int(coord[0]), int(coord[1])), int(radius[i]), c, -1)

    #     # plt.scatter(coord[0], coord[1],  zorder=2)
   
    

    # valid_xys =coords[valid_indices]
    # valid_radii = radius[valid_indices]
    # valid_num_tiles_hit = num_tile_hit[valid_indices]

    # # 绘制三维表面图
    # fig = plt.figure(figsize=(10, 7))
    # ax = fig.add_subplot(111, projection='3d')
    # x = np.arange(0, W)
    # y = np.arange(0, H)
    # x, y = np.meshgrid(x, y)
    #     # Z 值是高度，即 tensor 的第三维
    # surf = ax.plot_surface(x, y, per_pix_gs_n, cmap='viridis')
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    # # 设置标题和标签
    # ax.set_title('the number of gs per pixel')
    # ax.set_xlabel('X axis')
    # ax.set_ylabel('Y axis')
    # ax.set_zlabel('Height (Z axis)')
    # ax.set_zlabel('Height (Z axis)')
   
    # plt.savefig(f'{log_dir}/gs_nums_per_pix_{init_num_points}.png')
    plt.figure(figsize=(10, 7))
    plt.imshow(per_pix_gs_n)
    plt.colorbar()
    plt.title("the number of gs per pixel")
    plt.savefig(f"{log_dir}/gs_nums_per_pix_color_N{init_num_points}_I{iter}.png")

   