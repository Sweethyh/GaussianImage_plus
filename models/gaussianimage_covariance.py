import sys
from gsplat.project_gaussians_2d_covariance import project_gaussians_2d_covariance
from gsplat.rasterize_sum_plus import rasterize_gaussians_plus

from utils import *
import torch
import torch.nn as nn
import numpy as np
import math
from quantize import *
from optimizer import Adan

from models.utils import *


class GaussianImage_Covariance(nn.Module):
    def __init__(self, loss_type="L2", **kwargs):
        super().__init__()
        self.loss_type = loss_type
        self.init_num_points = kwargs["num_points"]
        self.cur_num_points = self.init_num_points
        self.H, self.W = int(kwargs["H"]), int(kwargs["W"])
        self.BLOCK_W, self.BLOCK_H = kwargs["BLOCK_W"], kwargs["BLOCK_H"]
        self.tile_bounds = (
            (self.W + self.BLOCK_W - 1) // self.BLOCK_W,
            (self.H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        )  #
        self.device = kwargs["device"]
        self.SLV = kwargs["args"].SLV_init
        self.color_norm = kwargs["args"].color_norm
        self.quantize = kwargs["quantize"]

        self.coords_norm = kwargs["args"].coords_norm
        self.coords_act = kwargs["args"].coords_act
        self.iterations = kwargs["args"].iterations
        self.gs_clip_coe = kwargs["args"].clip_coe
        self.radius_clip = kwargs["args"].radius_clip

        self.cov_quant = kwargs["args"].cov_quant
        self.color_quant = kwargs["args"].color_quant
        self.xy_quant = kwargs["args"].xy_quant
        self.logwriter = kwargs['logwriter']
        self.fake_quantizer = FakeQuantizationHalf.apply
        self.origin_bit = self.cur_num_points * 8 * 32

        # quantization parameters==============
        self.xy_bit = kwargs["args"].xy_bit
        self.cov_bit = kwargs["args"].cov_bit
        self.color_bit = kwargs["args"].color_bit

        w_init = torch.rand(self.init_num_points, 1, device=self.device) * self.W  # 0-w
        h_init = torch.rand(self.init_num_points, 1, device=self.device) * self.H  # 0-h
        self._xyz = nn.Parameter(torch.cat((w_init, h_init), dim=1))
        self.coords_activation = lambda x: x

        self._cov2d = nn.Parameter(torch.rand((self.init_num_points, 3), device=self.device))
        self.register_buffer('_opacity', torch.ones((self.init_num_points, 1)))
        self._features_dc = nn.Parameter(torch.zeros(self.init_num_points, 3, device=self.device))  # LIG (better)

        self.last_size = (self.H, self.W)
        self.register_buffer('background', torch.ones(3))
        if self.SLV:  # better in most cases
            low_pass = min(self.H * self.W / (9 * torch.pi * self.cur_num_points), 300)
            self.cholesky_bound = torch.tensor([low_pass, 0, low_pass]).view(1, 3).repeat(self.cur_num_points, 1).to(
                self.device)

        else:
            self.register_buffer('cholesky_bound', torch.tensor([0.5, 0, 0.5]).view(1, 3))
        self.register_buffer('bound', torch.tensor([0.5, 0.5]).view(1, 2))

        # activation fun============
        self.opacity_activation = lambda x: x
        self.color_activation = torch.sigmoid if self.color_norm else lambda x: x
        self.bkcolor_activation = lambda x: x

        self.rec_image = None  # 不同stage重建的图像
        # self.score = None
        self.quantized_cov2d = None
        self.xyz_quantizer = None
        self.xyz_quantizer_optimizer = None
        self.cholesky_quantizer = None
        self.cov2d_quantizer_optimizer = None
        self.features_dc_quantizer = None
        self.color_quantizer_optimizer = None
        self.xyz_scheduler = None
        self.cov2d_scheduler = None
        self.color_scheduler = None
        self.add_stage = 0

        if kwargs["opt_type"] == "adam":
            l = [
                {'params': [self._xyz], 'lr': kwargs["lr"], "name": "xyz"},
                {'params': [self._features_dc], 'lr': kwargs["lr"], "name": "f_dc"},
                {'params': [self._cov2d], 'lr': kwargs["lr"], "name": "cov2d"}
            ]

            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        else:
            self.optimizer = Adan(self.parameters(), lr=kwargs["lr"])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20000, gamma=0.5)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=70000, gamma=0.7)
        self.training_setup(lr=kwargs["lr"], quantize=self.quantize)

    def training_setup(self, lr, update_optimizer=False, quantize=False):
        if update_optimizer:
            l = [
                {'params': [self._xyz], 'lr': lr, "name": "xyz"},
                {'params': [self._features_dc], 'lr': lr, "name": "f_dc"},
                {'params': [self._cov2d], 'lr': lr, "name": "cov2d"}
            ]

            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20000, gamma=0.5)

        if quantize:
            lr = 0.001
            self.quantize = True
            if self.xy_quant == 'lsq':
                self.xyz_quantizer = UniformQuantizer(signed=False, bits=self.xy_bit, weight=1.0,
                                                      learned=True, num_channels=2).to(self.device)
                self.xyz_quantizer_optimizer = torch.optim.Adam(self.xyz_quantizer.parameters(), lr=lr)
                self.xyz_scheduler = torch.optim.lr_scheduler.StepLR(self.xyz_quantizer_optimizer, step_size=10000,
                                                                     gamma=0.5)
            else:
                self.xyz_quantizer = FakeQuantizationHalf.apply

            self.cholesky_quantizer = HybirdQuant(signed=False, bits=self.cov_bit, cov_bits=self.cov_bit,
                                                  learned=True, weight=1.0,
                                                  ).to(self.device)
            self.cov2d_quantizer_optimizer = torch.optim.Adam(self.cholesky_quantizer.parameters(), lr=lr,
                                                              eps=1e-15)
            self.cov2d_scheduler = torch.optim.lr_scheduler.StepLR(self.cov2d_quantizer_optimizer, step_size=10000,
                                                                   gamma=0.5)

            if self.color_quant == 'vq':
                self.features_dc_quantizer = VectorQuantizer(codebook_dim=3, codebook_size=8, num_quantizers=2,
                                                             vector_type="vector", kmeans_iters=5).to(self.device)
            elif self.color_quant == 'lsq':
                self.features_dc_quantizer = UniformQuantizer(signed=False, bits=self.color_bit, learned=True,
                                                              weight=1.0,
                                                              num_channels=3).to(self.device)
                self.color_quantizer_optimizer = torch.optim.Adam(self.features_dc_quantizer.parameters(), lr=lr,
                                                                  eps=1e-15)
                self.color_scheduler = torch.optim.lr_scheduler.StepLR(self.color_quantizer_optimizer, step_size=10000,
                                                                       gamma=0.5)

    def _init_data(self):
        self.cholesky_quantizer._init_data(self.get_cov2d_elements)
        if self.color_quant != 'vq':
            self.features_dc_quantizer._init_data(self.get_features)
        if self.xy_quant != 'fp16':
            self.xyz_quantizer._init_data(self._xyz)

    @property
    def get_xyz(self):
        return self.coords_activation(self._xyz)

    @property
    def get_features(self):
        return self.color_activation(self._features_dc)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_cov2d_elements(self):
        return self._cov2d + self.cholesky_bound  #

    @property
    def get_cov2d(self):
        return self._cov2d

    @property
    def get_std(self):
        covs = self.get_cov2d_elements
        std = torch.clamp((torch.cat([covs[:, 0:1], covs[:, 2:3]], dim=1)), 1e5, 300)
        return std

    def get_attributes(self):
        coords = self.xys.detach().clone().cpu().numpy()
        covs = self.get_cov2d_elements.detach().clone().cpu().numpy()
        colors = self.get_features.detach().clone().cpu().numpy()
        return {'coords': coords, 'covs': covs, 'colors': colors}

    def forward(self, isprint=False, H=None, W=None):
        tile_bounds = (
            (W + self.BLOCK_W - 1) // self.BLOCK_W,
            (H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        )

        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d_covariance(self.get_xyz,
                                                                                              self.get_cov2d_elements,
                                                                                              H, W, tile_bounds,
                                                                                              coords_norm=self.coords_norm,
                                                                                              clip_coe=self.gs_clip_coe,
                                                                                              radius_clip=self.radius_clip,
                                                                                              isprint=isprint,
                                                                                              )

        out_img = rasterize_gaussians_plus(self.xys, depths, self.radii, conics, num_tiles_hit,
                                           self.get_features, self.get_opacity, H, W, self.BLOCK_H, self.BLOCK_W,
                                           background=self.background,
                                           isprint=isprint,
                                           radius_clip=self.radius_clip
                                           )

        out_img = torch.clamp(out_img, 0, 1)  # [H, W, 3]
        out_img = out_img.view(-1, H, W, 3).permute(0, 3, 1, 2).contiguous()

        return {"render": out_img,
                "num_tiles_hit": num_tiles_hit,
                "radiii": self.radii,
                "visibility_filter": self.radii > 0,
                }

    def train_iter_quantize(self, gt_image):
        render_pkg = self.forward_quantize()
        image = render_pkg["render"]
        img_loss = loss_fn(image, gt_image, self.loss_type, lambda_value=0.7)

        loss = img_loss
        loss.backward()
        self.optimizer_step()

        with torch.no_grad():
            mse_loss = F.mse_loss(image, gt_image)
            psnr = 10 * math.log10(1.0 / mse_loss.item())

        return image.clone().detach(), loss, img_loss.item(), render_pkg["vq_loss"], psnr

    def optimizer_step(self):
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        if self.quantize:
            self.cov2d_quantizer_optimizer.step()
            self.cov2d_quantizer_optimizer.zero_grad()
            self.cov2d_scheduler.step()
            self.xyz_quantizer_optimizer.step()
            self.xyz_quantizer_optimizer.zero_grad()
            self.xyz_scheduler.step()
            self.color_quantizer_optimizer.step()
            self.color_quantizer_optimizer.zero_grad()
            self.color_scheduler.step()

    def train_iter(self, H, W, gt_image, isprint=False):
        render_pkg = self.forward(isprint=isprint, H=H, W=W)
        image = render_pkg["render"]
        loss = loss_fn(image, gt_image, self.loss_type, lambda_value=0.7)

        loss.backward()
        with torch.no_grad():
            mse_loss = F.mse_loss(image, gt_image)
            psnr = 10 * math.log10(1.0 / mse_loss.item())
        self.optimizer_step()
        return loss, psnr, image.detach()

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                    dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                                                       dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def update_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.zeros_like(extension_tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(extension_tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(extension_tensor).requires_grad_(True)
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(extension_tensor).requires_grad_(True)
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_cov2d, new_opacities=None, new_bkcolor=None):
        #  排除本身不正定的点
        none_definite, valid_mask = self.check_non_semi_definite(new_cov2d)

        d = {"xyz": new_xyz[valid_mask],
             "f_dc": new_features_dc[valid_mask],
             "cov2d": new_cov2d[valid_mask],
             }

        original_points_nums = self.cur_num_points
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._cov2d = optimizable_tensors["cov2d"]
        new_num_points = self._xyz.shape[0]
        self.cur_num_points = new_num_points
        self.add_stage += 1

        self._opacity = nn.Parameter(torch.ones((new_num_points, 1), device=self.device), requires_grad=False)

        if self.SLV:
            low_pass = min(self.H * self.W / (9 * torch.pi * self.cur_num_points), 300)
            # low_pass=self.filter_scheduler(self.add_stage)
            new_cholesky_bound = torch.tensor([low_pass, 0, low_pass]).view(1, 3).repeat(
                self.cur_num_points - original_points_nums, 1).to(self.device)
            self.cholesky_bound = torch.cat((self.cholesky_bound, new_cholesky_bound), dim=0)

        return new_num_points, none_definite

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def non_semi_definite_prune(self, H, W, cov2d=None):  # ltt
        to_prune_nums, valid_points_mask = self.check_non_semi_definite(cov2d=cov2d)

        if to_prune_nums and self.cur_num_points - to_prune_nums > 0:
            optimizable_tensors = self._prune_optimizer(valid_points_mask)
            self._xyz = optimizable_tensors["xyz"]
            self._features_dc = optimizable_tensors["f_dc"]
            new_num_points = self._xyz.shape[0]

            self._opacity = nn.Parameter(torch.ones((new_num_points, 1), device=self.device), requires_grad=False)
            self._cov2d = optimizable_tensors["cov2d"]
            if self.SLV:
                self.cholesky_bound = self.cholesky_bound[valid_points_mask]

            torch.cuda.empty_cache()
        pruned_num_points = self._xyz.shape[0]
        self.cur_num_points = pruned_num_points
        return to_prune_nums, pruned_num_points

    def check_non_semi_definite(self, cov2d=None):
        if cov2d is None:
            cov2d = self.get_cov2d_elements
        #  there we also exclude singular matrix
        valid_points_mask = torch.logical_and(cov2d[:, 0] * cov2d[:, 2] - cov2d[:, 1] ** 2 > 0,
                                              torch.logical_and(cov2d[:, 0] > 0, cov2d[:, 2] > 0))
        prune_mask = ~valid_points_mask
        to_prune_nums = torch.sum(prune_mask).item()

        return to_prune_nums, valid_points_mask

    def forward_quantize(self):
        l_vqm = 0
        means, l_vqm, m_bit, code_xy = self.xyz_quantizer(self._xyz)  # better
        means = self.coords_activation(means)

        # COV quantization==================
        cholesky_elements, l_vqs, s_bit, code = self.cholesky_quantizer(self.get_cov2d_elements)
        self.quantized_cov2d = cholesky_elements
        colors, l_vqc, c_bit, code_color = self.features_dc_quantizer(
            self.get_features)  # better if directly quantize the color after activation fun

        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d_covariance(means,
                                                                                              cholesky_elements, self.H,
                                                                                              self.W, self.tile_bounds,
                                                                                              clip_coe=self.gs_clip_coe,
                                                                                              radius_clip=self.radius_clip,
                                                                                              coords_norm=self.coords_norm
                                                                                              )
        out_img = rasterize_gaussians_plus(self.xys, depths, self.radii, conics, num_tiles_hit,
                                           colors, self.get_opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W,
                                           background=self.background)
        out_img = torch.clamp(out_img, 0, 1)
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        vq_loss = (l_vqm + l_vqs + l_vqc)

        return {"render": out_img, "vq_loss": vq_loss, "unit_bit": [m_bit, s_bit, c_bit]
                }

    def compress_wo_ec(self):
        if self.xy_quant == 'fp16':
            means = self.xyz_quantizer(self._xyz)  # fp16 28.1148
            quant_means = means
        else:
            means, quant_means = self.xyz_quantizer.compress(self._xyz)


        cholesky_elements, quant_cholesky_elements = self.cholesky_quantizer.compress(self.get_cov2d_elements)

        colors, feature_dc_index = self.features_dc_quantizer.compress(self.get_features)
        to_prune_nums, valid_points_mask = self.check_non_semi_definite(cholesky_elements)
        # remove the points if they are not semi-definite after quantization
        if to_prune_nums:
            cholesky_elements = cholesky_elements[valid_points_mask]
            quant_cholesky_elements = quant_cholesky_elements[valid_points_mask]
            means = means[valid_points_mask]
            feature_dc_index = feature_dc_index[valid_points_mask]
            colors = colors[valid_points_mask]
            new_num_points = valid_points_mask.sum().item()
            self._opacity = nn.Parameter(torch.ones((new_num_points, 1), device=self.device))
            if self.SLV:
                self.cholesky_bound = self.cholesky_bound[valid_points_mask]
            torch.cuda.empty_cache()
            self.cur_num_points = new_num_points
            # self.logwriter.write(f"to_prune_nums due to non-semi definite after quantization:{to_prune_nums} cur_num_points:{new_num_points}")
        # colors, feature_dc_index = self.features_dc_quantizer.compress(self.get_features[valid_points_mask])
        # return {"xyz":self._xyz.half(), "feature_dc_index": feature_dc_index, "quant_cholesky_elements": quant_cholesky_elements,}
        self.quantized_cov2d = cholesky_elements
        self.cur_num_points -= to_prune_nums
        return {"xyz": means, "feature_dc_index": feature_dc_index, "quant_cholesky_elements": quant_cholesky_elements,
                "quant_means": quant_means}

    def decompress_wo_ec(self, encoding_dict):
        xyz, feature_dc_index, quant_cholesky_elements = encoding_dict["xyz"], encoding_dict["feature_dc_index"], \
            encoding_dict["quant_cholesky_elements"]
        means = self.coords_activation(xyz)
        cholesky_elements = self.cholesky_quantizer.decompress(quant_cholesky_elements)
        colors = self.features_dc_quantizer.decompress(feature_dc_index)
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d_covariance(means,
                                                                                              cholesky_elements, self.H,
                                                                                              self.W, self.tile_bounds,
                                                                                              clip_coe=self.gs_clip_coe,
                                                                                              radius_clip=self.radius_clip,
                                                                                              coords_norm=self.coords_norm
                                                                                              )

        #
        out_img = rasterize_gaussians_plus(self.xys, depths, self.radii, conics, num_tiles_hit, colors,
                                           self.get_opacity, self.H, self.W,
                                           self.BLOCK_H, self.BLOCK_W,
                                           radius_clip=self.radius_clip)

        out_img = torch.clamp(out_img, 0, 1)
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        return {"render": out_img}

    def analysis_wo_ec(self, encoding_dict):
        feature_dc_index = encoding_dict["feature_dc_index"]
        quant_cholesky_elements = encoding_dict["quant_cholesky_elements"]

        total_bits = 0
        initial_bits, codebook_bits = 0, 0
        cholesky_bits, feature_dc_bits = 0, 0

        if self.cov_quant == "vq":
            for quantizer_index, layer in enumerate(self.cholesky_quantizer.quantizer.layers):
                cholesky_bits += layer._codebook.embed.numel() * torch.finfo(layer._codebook.embed.dtype).bits
            cov_index = quant_cholesky_elements.int().cpu().numpy()
            index_max = np.max(cov_index)
            max_bit = np.ceil(np.log2(index_max))
            cholesky_bits += cov_index.size * max_bit
        elif self.cov_quant == "lsq":
            cholesky_bits += quant_cholesky_elements.cpu().numpy().size * self.cholesky_quantizer.size() + 32 * 3 * 2

        if self.color_quant == "vq":
            for quantizer_index, layer in enumerate(self.features_dc_quantizer.quantizer.layers):
                codebook_bits += layer._codebook.embed.numel() * torch.finfo(layer._codebook.embed.dtype).bits
            feature_dc_index = feature_dc_index.int().cpu().numpy()
            index_max = np.max(feature_dc_index)
            max_bit = np.ceil(np.log2(index_max))  # calculate max bit for feature_dc_index
            feature_dc_bits += feature_dc_index.size * max_bit + codebook_bits
        elif self.color_quant == "lsq":
            feature_dc_bits += feature_dc_index.cpu().numpy().size * self.features_dc_quantizer.size() + 32 * 3 * 2

        if self.xy_quant == 'lsq':
            position_bits = encoding_dict['xyz'].cpu().numpy().size * self.xyz_quantizer.size() + 32 * 2 * 2
        elif self.xy_quant == "fp16":
            position_bits = self._xyz.numel() * 16

        total_bits += position_bits + cholesky_bits + feature_dc_bits

        bpp = total_bits / self.H / self.W
        position_bpp = position_bits / self.H / self.W
        cholesky_bpp = cholesky_bits / self.H / self.W
        feature_dc_bpp = feature_dc_bits / self.H / self.W
        return {"bpp": bpp, "position_bpp": position_bpp,
                "cholesky_bpp": cholesky_bpp, "feature_dc_bpp": feature_dc_bpp}
