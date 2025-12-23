from vector_quantize_pytorch import VectorQuantize, ResidualVQ
import torch
from torch import nn
import math
# from utils import *
import torch.nn.functional as F
from utils import compress_matrix_flatten_categorical, decompress_matrix_flatten_categorical, get_np_size
import numpy as np


def myabs(x):
    return torch.where(x == 0, x, torch.abs(x))


def mysign(x):
    return torch.where(x == 0, torch.ones_like(x), torch.sign(x))


def grad_scale(x, scale):
    return (x - x * scale).detach() + x * scale


def ste(x):
    return (x.round() - x).detach() + x


class FakeQuantizationHalf(torch.autograd.Function):
    """performs fake quantization for half precision"""

    @staticmethod
    def forward(_, x):
        return x.half().float()

    @staticmethod
    def backward(_, grad_output):
        return grad_output


class UniformQuantizer(nn.Module):
    '''
        LSQ +
    '''

    def __init__(self, signed=False, bits=8, learned=False, num_channels=1, entropy_type="none", weight=0.0001):
        super().__init__()
        self.bits = bits
        self.init_state = 0
        self.batch_init = 20
        if signed:
            self.qmin = -2 ** (bits - 1)
            self.qm = 2 ** (bits - 1) - 1
            self.qmax = 2 ** (bits - 1) - 1

        else:
            self.qmin = 0
            self.qmax = 2 ** bits - 1
            self.neg_min = -2 ** (bits - 1)
            self.neg_max = 2 ** (bits - 1)

            self.qm = 2 ** bits - 1

        self.learned = learned
        self.entropy_type = entropy_type
        if self.learned:
            self.scale = nn.Parameter(torch.ones(num_channels) / self.qmax, requires_grad=True)
            self.beta = nn.Parameter(torch.ones(num_channels) / self.qmax, requires_grad=True)
            # self.scale = nn.Parameter(torch.ones(2)/self.qmax, requires_grad=True)
            # self.beta = nn.Parameter(torch.ones(2)/self.qmax, requires_grad=True)
            # self.cov_scale = nn.Parameter(torch.ones(1)/self.neg_max, requires_grad=True)
            # self.cov_beta = nn.Parameter(torch.ones(1)/self.neg_max, requires_grad=True)

    def _init_data(self, tensor):
        device = tensor.device
        t_min, t_max = tensor.min(dim=0)[0], tensor.max(dim=0)[0]
        scale = (t_max - t_min) / (self.qmax - self.qmin)
        # cov_scale = (t_max - t_min) / (self.neg_max-self.neg_min)
        # self.beta.data = t_min.to(device)
        self.beta.data = t_min - self.qmin * scale.to(device)

        self.scale.data = scale.to(device)

        # self.beta.data = nn.Parameter(torch.Tensor([t_min[0],t_min[2]]).to(device))
        # self.scale.data = nn.Parameter(torch.Tensor([scale[0],scale[2]]).to(device))
        # self.cov_beta= nn.Parameter(t_min[1].to(device))
        # self.cov_scale.data = nn.Parameter(cov_scale [1].to(device))

    def get_params_orig(self):

        qm_clamped = torch.clamp(self.qm, min=1, max=self.qmax)

        # d_clamped = torch.clamp(self.scale, min=self.d_min, max=self.d_max)
        # d_clamped = torch.min(d_clamped, qm_clamped)

        # d_hard = 2 ** torch.round(torch.log2(d_clamped))
        # # if self.integer_bits_constraint:
        # qm_hard = d_hard * (2 ** (torch.floor(torch.log2(torch.ceil(qm_clamped / d_hard))) + 1) - 1)
        # else:
        qm_hard = 2 ** (torch.floor(torch.log2(qm_clamped)) + 1) - 1
        # d_hard = torch.min(d_hard, qm_hard)

        qm_ste = (qm_hard - qm_clamped).detach() + qm_clamped
        # self.qmax=qm_hard
        # d_ste = (d_hard - d_clamped).detach() + d_clamped

        return {
            # 'd_clamped': d_clamped, 
            'qm_clamped': qm_clamped,
            # 'd_hard': d_hard, 'd_ste': d_ste,
            'qm_ste': qm_ste,
            'qm_hard': qm_hard}

    def get_bits_diff(self, params_dict):

        qm_ste = params_dict['qm_ste']

        bits_hard = torch.ceil(torch.log2(qm_ste + 1) + 1)
        bits_smooth = torch.log2(qm_ste + 1) + 1

        # I think that bits_smooth should be additionally clamped ( min 2, because then there is an incentive to regularize grid with 2 bits and that should not happen)
        bits_hard = torch.clamp(bits_hard, min=2.)
        bits_smooth = torch.clamp(bits_smooth, min=2.)

        return (bits_hard - bits_smooth).detach() + bits_smooth

    def forward(self, x, quant_loss=False):
        bits, entropy_loss = 0, 0

        if self.init_state == 0:
            self._init_data(x)
            self.init_state += 1
        # if self.learned:
        grad = 1.0 / ((self.qmax * x.numel()) ** 0.5)  # 在 S 的梯度上还乘了一个缩放系数
        s_scale = grad_scale(self.scale, grad)
        beta_scale = grad_scale(self.beta, grad)
        s_scale, beta_scale = self.scale, self.beta
        code = ((x - beta_scale) / s_scale).clamp(self.qmin, self.qm)
        # code =torch.clamp( ((x - beta_scale) / s_scale),min=self.qmin)

        quant = ste(code)
        dequant = quant * s_scale + beta_scale
        return dequant, entropy_loss, bits, quant

    def size(self):
        return self.bits

    def reset_state(self):
        self.init_state = 0

    def compress(self, x):
        code = ((x - self.beta) / self.scale).clamp(self.qmin, self.qmax)
        # code=torch.clamp(  ((x - self.beta) / self.scale),min=self.qmin)
        return code.round() * self.scale + self.beta, code.round()

    def decompress(self, x):
        return x * self.scale + self.beta


class LogQuantizer(nn.Module):
    '''
        log quantizer
    '''

    def __init__(self, signed=True, bits=8, learned=False, num_channels=1, entropy_type="none",
                 weight=0.001):
        super().__init__()
        self.bits = bits
        self.init_state = 0

        if signed:
            self.qmin = -2 ** (bits - 1)
            self.qmax = 2 ** (bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** bits - 1
            self.neg_min = -2 ** (bits - 1)
            self.neg_max = 2 ** (bits - 1)

        self.learned = learned
        self.entropy_type = entropy_type
        if self.learned:
            self.scale = nn.Parameter(torch.ones(num_channels, device=torch.device('cuda')) / self.qmax,
                                      requires_grad=True)
            self.beta = nn.Parameter(torch.ones(num_channels, device=torch.device('cuda')) / self.qmax,
                                     requires_grad=True)
        else:
            self.beta = torch.empty((num_channels))  # ,device=torch.device('cuda')))
            self.scale = torch.empty(num_channels)  # ,device=torch.device('cuda')
        self.weight = weight
        self.min_log, self.max_log = 0, 0
        self.sign = None

    def _init_data(self, tensor):
        device = tensor.device
        tensor = torch.log(torch.abs(tensor) + 1e-6)
        t_min, t_max = tensor.min(dim=0)[0], tensor.max(dim=0)[0]
        scale = (t_max - t_min) / (self.qmax - self.qmin)
        self.scale.data = scale.to(device)
        self.beta.data = t_min.to(device)
        # cov_scale = (t_max - t_min) / (self.neg_max-self.neg_min)
        self.min_log = t_min.to(device)
        self.max_log = t_max.to(device)

    def forward(self, x, quant_loss=False):
        bits, entropy_loss = 0, 0
        if self.init_state == 0:
            self._init_data(x)
            self.init_state += 1
        if self.learned:
            grad = 1.0 / ((self.qmax * x.numel()) ** 0.5)
            s_scale = grad_scale(self.scale, grad)
            beta_scale = grad_scale(self.beta, grad)
            s_scale, beta_scale = self.scale, self.beta
            # 计算张量的绝对值并取对数
            log_tensor = torch.log(torch.abs(x) + 1e-6)  # 加上1e-6以防止取对数时出现零 # 线性量化
            code = ((log_tensor - beta_scale) / s_scale).clamp(self.qmin, self.qmax)
            quant = ste(code)
            dequantized_log_tensor = quant * s_scale + beta_scale
            dequant = torch.sign(x) * torch.exp(dequantized_log_tensor)
        else:
            # 计算张量的绝对值并取对数 
            log_tensor = torch.log(torch.abs(x) + 1e-6)  # 加上1e-6以防止取对数时出现零 # 线性量化
            # self.min_log = torch.min(log_tensor) 
            self.beta = torch.min(log_tensor)

            self.max_log = torch.max(log_tensor)
            # scale = (self.qmax - self.qmin) / (self.max_log - self.min_log) 
            self.scale = (self.max_log - self.beta) / (self.qmax - self.qmin)
            code = ((log_tensor - self.beta) / self.scale).clamp(self.qmin, self.qmax)
            quant = ste(code)
            # 反量化 
            dequantized_log_tensor = quant * self.scale + self.beta
            # dequant = torch.sign(x) * torch.exp(dequantized_log_tensor)
            dequant = torch.exp(dequantized_log_tensor)

        return dequant, entropy_loss, bits, quant

    def size(self):
        return self.bits

    def reset_state(self):
        self.init_state = 0

    def compress(self, x):
        if not self.learned:
            self._init_data(x)
        log_tensor = torch.log(torch.abs(x) + 1e-6)
        # min_log = torch.min(log_tensor) 
        # max_log = torch.max(log_tensor) 
        # scale = (self.max_log - self.min_log)  / (self.qmax - self.qmin)
        # print(log_tensor.device, self.beta.device)
        code = ((log_tensor - self.beta) / self.scale).clamp(self.qmin, self.qmax)
        self.sign = torch.sign(x)
        # return  torch.sign(x) * torch.exp(code.round()* self.scale +  self.beta), code.round()
        return torch.exp(code.round() * self.scale + self.beta), code.round()

    def decompress(self, x):
        # scale = (self.max_log - self.min_log)  / (self.qmax - self.qmin)
        return torch.exp(x * self.scale + self.beta)


class VectorQuantizer(nn.Module):
    def __init__(self, num_quantizers=1, codebook_dim=1, codebook_size=64, kmeans_iters=10, vector_type="vector"):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.vector_type = vector_type
        if self.num_quantizers == 1:
            if self.vector_type == "vector":
                self.quantizer = VectorQuantize(dim=codebook_dim, codebook_size=codebook_size,
                                                decay=0.8, commitment_weight=1., kmeans_init=True,
                                                kmeans_iters=kmeans_iters)
        else:
            if self.vector_type == "vector":
                self.quantizer = ResidualVQ(dim=codebook_dim, codebook_size=codebook_size,
                                            num_quantizers=num_quantizers,
                                            decay=0.8, commitment_weight=1.,
                                            kmeans_init=True, kmeans_iters=kmeans_iters)

    def forward(self, x):
        if self.training:
            x, _, l_vq = self.quantizer(x)
            l_vq = torch.sum(l_vq)
            return x, l_vq, 0
        else:
            num_points, num_channels = x.shape
            x, embed_index, l_vq = self.quantizer(x)
            l_vq = torch.sum(l_vq)
            bits = self.size(embed_index)
            # unit_bit = bits / num_points / num_channels
            return x, l_vq, bits

    def size(self, embed_index):
        if self.num_quantizers == 1:
            if self.vector_type == "vector":
                codebook_bits = self.quantizer._codebook.embed.numel() * torch.finfo(
                    self.quantizer._codebook.embed.dtype).bits
            elif self.vector_type == "ste":
                codebook_bits = self.quantizer.embedding.weight.data.numel() * torch.finfo(
                    self.quantizer.embedding.weight.data.dtype).bits
            index_bits = 0
            compressed, histogram_table, unique = compress_matrix_flatten_categorical(
                embed_index.int().flatten().tolist())
            index_bits += get_np_size(compressed) * 8
            index_bits += get_np_size(histogram_table) * 8
            index_bits += get_np_size(unique) * 8
        else:
            codebook_bits, index_bits = 0, 0
            for quantizer_index, layer in enumerate(self.quantizer.layers):
                if self.vector_type == "vector":
                    codebook_bits += layer._codebook.embed.numel() * torch.finfo(layer._codebook.embed.dtype).bits
                elif self.vector_type == "ste":
                    codebook_bits += layer.embedding.weight.data.numel() * torch.finfo(
                        layer.embedding.weight.data.dtype).bits
            compressed, histogram_table, unique = compress_matrix_flatten_categorical(
                embed_index.int().flatten().tolist())
            index_bits += get_np_size(compressed) * 8
            index_bits += get_np_size(histogram_table) * 8
            index_bits += get_np_size(unique) * 8
        total_bits = codebook_bits + index_bits
        # print("vq:", embed_index.shape, codebook_bits, index_bits)
        return total_bits

    def compress(self, x):
        x, embed_index, _ = self.quantizer(x)
        return x, embed_index

    def decompress(self, embed_index):
        recon = 0
        if hasattr(self.quantizer, 'layers'):
            for i, layer in enumerate(self.quantizer.layers):
                recon += layer._codebook.embed[0, embed_index[:, i]]
        else:
            recon = self.quantizer._codebook.embed[0, embed_index]
        return recon


class HybirdQuant(nn.Module):
    def __init__(self, signed=False, bits=8, cov_bits=10, learned=False, num_channels=1, entropy_type="none",
                 weight=0.001):
        super(HybirdQuant, self).__init__()

        # # var===========
        self.init_state = 0

        self.var_quantizer = LogQuantizer(False, bits, learned=False, num_channels=2,
                                          entropy_type=entropy_type, weight=weight)
        self.cov_quantizer = UniformQuantizer(signed, cov_bits, learned=True, num_channels=1,
                                              entropy_type=entropy_type, weight=weight)

        self.bits = bits

    def _init_data(self, tensor):
        self.var_quantizer._init_data(tensor[:, ::2])
        self.cov_quantizer._init_data(tensor[:, 1:2])

    def forward(self, x, quant_loss=False):
        if self.init_state == 0:
            self._init_data(x)
            self.init_state += 1
        var = x[:, ::2]
        cov = x[:, 1:2]
        dequant_var, entropy_loss, bits, code_quant_var = self.var_quantizer(var, quant_loss)
        dequant_cov, entropy_loss_cov, bits_cov, code_quant_cov = self.cov_quantizer(cov, quant_loss)

        dequant = torch.cat([dequant_var[:, 0:1], dequant_cov, dequant_var[:, 1:2]], dim=1)
        code_quant = torch.cat([code_quant_var[:, 0:1], code_quant_cov, code_quant_var[:, 1:]], dim=1)
        return dequant, entropy_loss + entropy_loss_cov, bits + bits_cov, code_quant

    def size(self):
        return (self.cov_quantizer.size() + self.var_quantizer.size() * 2) / 3

    def reset_state(self):
        self.var_quantizer.reset_state()
        self.cov_quantizer.reset_state()

    def compress(self, x):
        var = x[:, ::2]
        cov = x[:, 1:2]
        # dequant_var,code_var =self.var_quantizer(var)
        dequant_var, code_var = self.var_quantizer.compress(var)
        dequant_cov, code_cov = self.cov_quantizer.compress(cov)
        return torch.cat([dequant_var[:, 0:1], dequant_cov, dequant_var[:, 1:2]], dim=1), torch.cat(
            [code_var[:, 0:1], code_cov, code_var[:, 1:]], dim=1)

    def decompress(self, x):
        var = x[:, ::2]
        cov = x[:, 1:2]
        cov = self.cov_quantizer.decompress(cov)
        var = self.var_quantizer.decompress(var)
        return torch.cat([var[:, 0:1], cov, var[:, 1:2]], dim=1)
