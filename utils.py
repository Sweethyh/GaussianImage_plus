import os
import numpy as np
import torch
from pathlib import Path
from PIL import Image,ImageOps
import torchvision.transforms as transforms
import constriction



def save_out_image(log_dir,out_image,image_name,dirname='output',postfix=''):
    out_dir =os.path.join(os.path.dirname(log_dir),dirname) if dirname is not None else os.path.dirname(log_dir)
    print("save into ",out_dir)
    transform = transforms.ToPILImage()
    os.makedirs(out_dir,exist_ok=True)
    img = transform(out_image.squeeze(0))
    name =os.path.join(out_dir,f"{image_name}.png") if postfix=='' else os.path.join(out_dir,f"{image_name}_{postfix}.png")
    img.save( name)


def image_path_to_tensor(image_path: Path):
    img = Image.open(image_path)
    # img_flip = ImageOps.mirror(img)
    transform = transforms.ToTensor()
    img_tensor = transform(img).unsqueeze(0) #[1, C, H, W]
    # img_flip=transform(img_flip).unsqueeze(0) #[1, C, H, W]
    return img_tensor




class LogWriter:
    def __init__(self, file_path, train=True):
        os.makedirs(file_path, exist_ok=True)
        self.file_path = os.path.join(file_path, "train.txt" if train else "test_v2.txt")

    def write(self, text):
        # 打印到控制台
        print(text)
        # 追加到文件
        with open(self.file_path, 'a') as file:
            file.write(text + '\n')


# codec=======================================
def judege_type(min, max):
    if min>=0:
        if max<=256:
            return np.uint8
        elif max<=65535:
            return np.uint16
        else:
            return np.uint32
    else:
        if max<128 and min>=-128:
            return np.int8
        elif max<32768 and min>=-32768:
            return np.int16
        else:
            return np.int32
def compress_matrix_flatten_categorical(matrix, return_table=False):
    '''
    :param matrix: np.array
    :return compressed, symtable
    '''
    matrix = np.array(matrix) #matrix.flatten()
    unique, unique_indices, unique_inverse, unique_counts = np.unique(matrix, return_index=True, return_inverse=True, return_counts=True, axis=None)
    min_value = np.min(unique)
    max_value = np.max(unique)
    unique = unique.astype(judege_type(min_value, max_value))
    message = unique_inverse.astype(np.int32)
    probabilities = unique_counts.astype(np.float64) / np.sum(unique_counts).astype(np.float64)
    entropy_model = constriction.stream.model.Categorical(probabilities)
    encoder = constriction.stream.stack.AnsCoder()
    encoder.encode_reverse(message, entropy_model)
    compressed = encoder.get_compressed()
    return compressed, unique_counts, unique

def decompress_matrix_flatten_categorical(compressed, unique_counts, quant_symbol, symbol_length, symbol_shape):
    '''
    :param matrix: np.array
    :return compressed, symtable
    '''
    probabilities = unique_counts.astype(np.float64) / np.sum(unique_counts).astype(np.float64)
    entropy_model = constriction.stream.model.Categorical(probabilities)
    decoder = constriction.stream.stack.AnsCoder(compressed)
    decoded = decoder.decode(entropy_model, symbol_length)
    decoded = quant_symbol[decoded].reshape(symbol_shape)#.astype(np.int32)
    return decoded

def get_np_size(x):
    return x.size * x.itemsize

def compress_matrix_flatten_gaussian_global(matrix):
    '''
    :param matrix: tensor
    :return compressed, symtable
    '''
    mean = torch.mean(matrix).item()
    std=torch.std(matrix).clamp(1e-5, 1e10).item()
    min_value, max_value = matrix.min().int().item(), matrix.max().int().item()
    if min_value == max_value:
        max_value = min_value + 1
    message = np.array(matrix.int().flatten().cpu().tolist(), dtype=np.int32)
    entropy_model = constriction.stream.model.QuantizedGaussian(min_value, max_value, mean, std)
    encoder = constriction.stream.stack.AnsCoder()
    encoder.encode_reverse(message, entropy_model)
    compressed = encoder.get_compressed()
    total_bits = get_np_size(compressed) * 8
    return total_bits








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


# general utils


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


# def check_non_semi_definite(cov2d):#ltt
#         if isinstance(cov2d, torch.Tensor):
#             non_defi = torch.sum(cov2d[:, 0] * cov2d[:, 2] - cov2d[:, 1] ** 2 <= 0)
#             valid_points_mask = torch.logical_and(
#                     cov2d[:, 0] * cov2d[:, 2] - cov2d[:, 1] ** 2 > 0,
#                     torch.logical_and(cov2d[:, 0] > 0, cov2d[:, 2] > 0)
#                 )
#             prune_mask = ~valid_points_mask
#             to_prune_nums = torch.sum(prune_mask)
#         else:
#             non_defi = np.sum(cov2d[:, 0] * cov2d[:, 2] - cov2d[:, 1] ** 2 <= 0)
#             valid_points_mask = np.logical_and(
#                         cov2d[:, 0] * cov2d[:, 2] - cov2d[:, 1] ** 2 > 0,
#                         np.logical_and(cov2d[:, 0] > 0, cov2d[:, 2] > 0)
#                     )
#             prune_mask = ~valid_points_mask
#             to_prune_nums = np.sum(prune_mask)
#         if to_prune_nums:
#             print(f"<=0 {non_defi} x&y<=0 {to_prune_nums-non_defi}")
#             # self.logwrit(f"<=0 {non_defi} x&y<=0 {to_prune_nums-non_defi}")
#         return to_prune_nums,valid_points_mask


   