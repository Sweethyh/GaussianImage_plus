from gsplat.project_gaussians_2d import project_gaussians_2d
from gsplat.rasterize_sum import rasterize_gaussians_sum

from utils import *
import torch
import torch.nn as nn
import numpy as np
import math
from quantize import *
from optimizer import Adan

class GaussianImage_Cholesky(nn.Module):
    def __init__(self, loss_type="L2", **kwargs):
        super().__init__()
        self.loss_type = loss_type
        self.init_num_points = kwargs["num_points"]
        self.cur_num_points =self.init_num_points 
        self.H, self.W = int(kwargs["H"]), int(kwargs["W"])
        self.BLOCK_W, self.BLOCK_H = kwargs["BLOCK_W"], kwargs["BLOCK_H"]
        self.tile_bounds = (
            (self.W + self.BLOCK_W - 1) // self.BLOCK_W,
            (self.H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        ) # 
        self.device = kwargs["device"]
        # self.lamda = kwargs["lamda"]
        self.mirror=kwargs["args"].camera=="mirror"
        self.radius_type = kwargs['args'].radius
        self.learnable_opacity = kwargs["args"].opacity
        self.learnable_back_color=kwargs["args"].backcolor
        self.low_opac_thred = kwargs["args"].low_opac_thred
        self.SLV=kwargs["args"].SLV_init
        self.color_norm=kwargs["args"].color_norm
        self.bkcolor_norm=kwargs["args"].bkcolor_norm
        self.quantize = kwargs["quantize"]
        self.alpha_blend=kwargs["args"].alpha_blend
        self.coords_norm=kwargs["args"].coords_norm
        # ==========
        self.max_sh_degree =  kwargs["args"].sh_degree
        # self._xyz = torch.empty(0)
        # self._features_dc = torch.empty(0)
        # self._features_rest = torch.empty(0)
        # self._scaling = torch.empty(0)
        # self._rotation = torch.empty(0)
        # self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        # self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        # self.optimizer =torch.empty(0)
        # self.scheduler = torch.empty(0)
        # ========


        self._xyz = nn.Parameter(torch.atanh(2 * (torch.rand(self.init_num_points, 2) - 0.5)))
        self._cholesky = nn.Parameter(torch.rand(self.init_num_points, 3))
        if self.learnable_opacity  :
            # self._opacity= nn.Parameter(torch.logit(0.1 * torch.ones(self.init_num_points, 1)))
            self._opacity= nn.Parameter(torch.rand(self.init_num_points, 1))
        else:
            self.register_buffer('_opacity', torch.ones((self.init_num_points, 1)))
        # self.register_buffer('_opacity', torch.ones((self.init_num_points, 1)))
        self._features_dc = nn.Parameter(torch.rand(self.init_num_points, 3)) # xinjie
        self.last_size = (self.H, self.W)
        if self.learnable_back_color:
            self.background =  nn.Parameter(torch.zeros(self.init_num_points, 3, device=self.device))
        else:
            self.register_buffer('background', torch.ones(3))
        # self.register_buffer('background', torch.ones(3))
        self.opacity_activation = torch.sigmoid if self.learnable_opacity else lambda x: x
        self.rgb_activation=torch.sigmoid if self.color_norm else lambda x: x
        self.bkcolor_activation=torch.sigmoid if self.bkcolor_norm else lambda x: x
        self.coords_activation=torch.tanh if self.coords_norm else lambda x: x

        # self.opacity_activation = torch.sigmoid
        # self.rgb_activation = torch.sigmoid
        self.register_buffer('bound', torch.tensor([0.5, 0.5]).view(1, 2))
        if self.SLV:
            low_pass=min(self.H*self.W/(9*torch.pi*self.cur_num_points),300)
            self.cholesky_bound=torch.tensor([low_pass,  0, low_pass]).view(1, 3).repeat(self.cur_num_points,1).to(self.device)
            #  self.cholesky_bound=torch.tensor([self.H*self.W/(9*torch.pi*self.init_num_points), 0, 
                                        #   self.H*self.W/(9*torch.pi*self.init_num_points)]).view(1, 3).to(self.device)
        else:
            self.register_buffer('cholesky_bound', torch.tensor([0.5, 0, 0.5]).view(1, 3))
        # self.register_buffer('cholesky_bound', torch.tensor([0.5, 0, 0.5]).view(1, 3))

        if self.quantize:
            self.xyz_quantizer = FakeQuantizationHalf.apply 
            self.features_dc_quantizer = VectorQuantizer(codebook_dim=3, codebook_size=8, num_quantizers=2, vector_type="vector", kmeans_iters=5) 
            self.cholesky_quantizer = UniformQuantizer(signed=False, bits=6, learned=True, num_channels=3)
            self.cholesky_optimizer =None
        if  kwargs["args"].camera!="mirror":
            self._xyz = nn.Parameter(torch.atanh(2 * (torch.rand(self.init_num_points, 2,device=self.device) - 0.5)))
            self._cholesky = nn.Parameter(torch.rand(self.init_num_points, 3,device=self.device))
            self.register_buffer('_opacity', torch.ones(self.init_num_points, 1,device=self.device))
            # self._features_dc = nn.Parameter(torch.rand(self.init_num_points, 3,device=self.device))
            self._features_dc = nn.Parameter(torch.zeros(self.init_num_points, 3,device=self.device))

            #  xinjie's ==============
            if kwargs["opt_type"] == "adam":
                if kwargs["args"].opt_nums>1: # 每组参数使用不同的优化器
                    l = [
                    {'params': [self._xyz], 'lr':  kwargs["args"].position_lr_init * self.spatial_lr_scale, "name": "xyz"},
                    {'params': [self._features_dc], 'lr': kwargs["args"].feature_lr, "name": "f_dc"},
                    # {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
                    # {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                    {'params': [self._cholesky], 'lr':  kwargs["args"].cholesky_lr, "name": "cholesky"},
                    ]
                    self.optimizer =   torch.optim.Adam(l, lr=0.0, eps=1e-15)
                else:
                    self.optimizer = torch.optim.Adam(self.parameters(), lr=kwargs["lr"])
            else:
                if kwargs["args"].opt_nums>1: # 每组参数使用不同的优化器
                    l = [
                    {'params': [self._xyz], 'lr': kwargs["lr"], "name": "xyz"},
                    {'params': [self._features_dc], 'lr': kwargs["lr"], "name": "f_dc"},
                    # {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
                    # {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                    {'params': [self._cholesky], 'lr':  kwargs["lr"], "name": "cholesky"},
                    # {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
                    ]
                    # if self.quantize:
                        # l.append({'params': [self.cholesky_quantizer.parameters() ], 'lr':  kwargs["lr"], "name": "cholesky_quantizer"})
                    
                    self.optimizer =  Adan(l, lr=0.0, eps=1e-15)
                else:
                    self.optimizer = Adan(self.parameters(), lr=kwargs["lr"])
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20000, gamma=0.5)

    def _init_data(self):
        self.cholesky_quantizer._init_data(self._cholesky)

    @property
    def get_xyz(self):
        return torch.tanh(self._xyz)
        # return self._xyz
    
    @property
    def get_features(self):
        return self.rgb_activation(self._features_dc)
    
    @property
    def get_opacity(self):
        return self._opacity

    @property
    def get_cholesky_elements(self):
        return self._cholesky+self.cholesky_bound
    def get_attributes(self):
        coords=self.xys.detach().clone().cpu().numpy()
        l = self.get_cholesky_elements.detach().clone()
        covs=torch.stack([l[:,0]*l[:,0],l[:,0]*l[:,1],l[:,1]*l[:,1]+l[:,2]*l[:,2]],dim=1).cpu().numpy()
        colors =  self.get_features.detach().clone().cpu().numpy()
        return {'coords': coords, 'covs': covs, 'colors': colors}
    def training_setup(self,lr, quantize=False,iter=0):
            if quantize:
                l = [
                    {'params': [self._xyz], 'lr':lr, "name": "xyz"},
                    {'params': [self._features_dc], 'lr': lr, "name": "f_dc"},
                    # {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
                    # {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                    {'params': [self._cholesky], 'lr': lr, "name": "cholesky"},
                    # {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
                    ]
                
                self.optimizer =  Adan(l, lr=0.0, eps=1e-15)
                self.cholesky_optimizer = torch.optim.Adam(self.cholesky_quantizer.parameters(), lr=lr)
                self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20000, gamma=0.5)
        # ltt
    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float=1):

        #  ori initialization===========
        self._xyz = nn.Parameter(torch.atanh(2 * (torch.rand(self.init_num_points, 2,device=self.device) - 0.5)))
        self._cholesky = nn.Parameter(torch.rand(self.init_num_points, 3,device=self.device))
        self.register_buffer('_opacity', torch.ones(self.init_num_points, 1,device=self.device))
        # self._features_dc = nn.Parameter(torch.rand(self.init_num_points, 3,device=self.device))
        self._features_dc = nn.Parameter(torch.zeros(self.init_num_points, 3,device=self.device))

# #       mirge initi===============
#         # self.spatial_lr_scale = spatial_lr_scale
#         fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda() # 5w,3
#         fused_point_cloud = torch.cat([fused_point_cloud[:,0:1],fused_point_cloud[:,2:3]], dim=1)
#         fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
#         features = torch.zeros(fused_color.shape[0], 3).float().cuda()
#         features = fused_color
#         # features[:, 3:, 1:] = 0.0

#         print("Number of points at initialisation : ", fused_point_cloud.shape[0])

#         # dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
#         # scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 2)
#         # rots = torch.ones((fused_point_cloud.shape[0], 4), device="cuda")
#         #rots[:, 1] = 1

#         # opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
#         self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
#         self._features_dc = nn.Parameter(features.requires_grad_(True))
#         self._cholesky = nn.Parameter(torch.rand(self.init_num_points, 3,device="cuda"))
#         # self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
#         # self._scaling = nn.Parameter(scales.requires_grad_(True))
#         # self._rotation = nn.Parameter(rots.requires_grad_(True))
#         # self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
    def forward(self,H,W,isprint=False, iter=0):
        screenspace_points = torch.zeros((self.get_xyz.shape[0],4), dtype=self.get_xyz.dtype, requires_grad=True, device=self.device) + 0
        self.xys,screenspace_points,depths,  self.radii, conics, num_tiles_hit = project_gaussians_2d(self.get_xyz, screenspace_points,
                            self.get_cholesky_elements,  self.H, self.W, self.tile_bounds,isprint=isprint)
        if self.alpha_blend:
            out_img,per_pix_gs_nums,screenspace_points,accum_weights, accum_max_weight,  accum_weights_count, accum_max_count = rasterize_gaussians_alpha_blending(self.xys,screenspace_points, depths, self.radii, conics, num_tiles_hit,
                    self.get_features, self.get_opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, 
                    return_alpha=False,
                    isprint=isprint
                    # isprint=True
                    )
        else:
            out_img,per_pix_gs_nums,  screenspace_points = rasterize_gaussians_sum(self.xys, screenspace_points,depths, self.radii, conics, num_tiles_hit,
                    self.get_features, self.get_opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
        out_img = torch.clamp(out_img, 0, 1) #[H, W, 3]
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
      
        return {"render": (out_img), "num_tiles_hit":num_tiles_hit,
                "radiii":self.radii,
                "per_pix_gs_nums":per_pix_gs_nums,
                  "visibility_filter" : self.radii > 0,
                "screen_points":screenspace_points}
    def render(self,H, W, mirror=False,isprint=False):
        screenspace_points = torch.zeros((self.get_xyz.shape[0],4), dtype=self.get_xyz.dtype, requires_grad=True, device=self.device) + 0
        tile_bounds = ( (W + self.BLOCK_W - 1) // self.BLOCK_W, (H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        ) #
        self.xys, screenspace_points,depths,  self.radii, conics, num_tiles_hit = project_gaussians_2d(self.get_xyz, 
            screenspace_points, self.get_cholesky_elements,  self.H, self.W, tile_bounds,isprint=isprint)
        if self.alpha_blend:
            out_img,per_pix_gs_nums,screenspace_points,accum_weights, accum_max_weight,  accum_weights_count, accum_max_count = rasterize_gaussians_alpha_blending(self.xys,screenspace_points, depths, self.radii, conics, num_tiles_hit,
                    self.get_features, self.get_opacity, H, W, self.BLOCK_H, self.BLOCK_W, background=self.background, 
                    return_alpha=False,
                    isprint=isprint
                    # isprint=True
                    )
        else:
            out_img,per_pix_gs_nums,  screenspace_points = rasterize_gaussians_sum(self.xys, screenspace_points,depths, self.radii, conics, num_tiles_hit,
                    self.get_features, self.get_opacity, H, W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
        out_img = torch.clamp(out_img, 0, 1) #[H, W, 3]
        out_img = out_img.view(-1, H, W, 3).permute(0, 3, 1, 2).contiguous()
      
        return {"render": (out_img), "num_tiles_hit":num_tiles_hit,
                "radiii":self.radii,
                "per_pix_gs_nums":per_pix_gs_nums,
                  "visibility_filter" : self.radii > 0,
                "screen_points":screenspace_points}
    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)
                if 'exp_avg_diff' in stored_state.keys():
                    stored_state['exp_avg_diff']=torch.cat((stored_state['exp_avg_diff'], torch.zeros_like(extension_tensor)), dim=0)
                    stored_state['neg_pre_grad']=torch.cat((stored_state['neg_pre_grad'], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
    def densification_postfix(self, new_xyz, new_features_dc,  new_cov2d,new_opacities=None,new_bkcolor=None):
        d = {"xyz": new_xyz,
            "f_dc": new_features_dc,
            "cholesky" : new_cov2d,
            }
        # if self.learnable_opacity:
        #     d["opacity"]= new_opacities
        # if self.learnable_back_color:
        #     d["bkcolor"]=new_bkcolor
        original_points_nums=self.cur_num_points
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._cholesky = optimizable_tensors["cholesky"]
        new_num_points=self._xyz.shape[0]
        self.cur_num_points=new_num_points
        self._opacity= nn.Parameter(torch.ones((new_num_points, 1), device=self.device),requires_grad=False)
        if self.SLV:
            low_pass=min(self.H*self.W/(9*torch.pi*self.cur_num_points),300)
            new_cholesky_bound=torch.tensor([low_pass,  0, low_pass]).view(1, 3).repeat(self.cur_num_points-original_points_nums,1).to(self.device)
            self.cholesky_bound=torch.cat((self.cholesky_bound, new_cholesky_bound), dim=0)
        return new_num_points,0
    def add_densification_stats(self, viewspace_point_tensor, update_filter):
            # self.xyz_gradient_accum[update_filter] += torch.norm(torch.abs(viewspace_point_tensor.grad[update_filter,:2]), dim=-1, keepdim=True)
            self.xyz_gradient_accum[update_filter] += torch.norm((viewspace_point_tensor.grad[update_filter,:2]), dim=-1, keepdim=True)
            self.denom[update_filter] += 1
    def train_iter(self, H,W,gt_image,isprint=False,iter=iter):
        render_pkg = self.forward(H,W,isprint=isprint)
        image = render_pkg["render"]
        loss = loss_fn(image, gt_image, self.loss_type, lambda_value=0.7)
        loss.backward()
        with torch.no_grad():
            mse_loss = F.mse_loss(image, gt_image)
            psnr = 10 * math.log10(1.0 / mse_loss.item())
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none = True)
        # if self.cholesky_optimizer is not None:
        #     self.cholesky_optimizer.zero_grad()
        #     self.cholesky_optimizer.step()
            
        self.scheduler.step()
        return loss, psnr,image.detach()

    def forward_quantize(self):
        l_vqm, m_bit = 0, 16*self.cur_num_points*2
        means = torch.tanh(self.xyz_quantizer(self._xyz))
        cholesky_elements, l_vqs, s_bit,_ = self.cholesky_quantizer(self._cholesky)
        cholesky_elements = cholesky_elements + self.cholesky_bound
        l_vqr, r_bit = 0, 0
        colors, l_vqc, c_bit = self.features_dc_quantizer(self.get_features)
        screenspace_points = torch.zeros((self.get_xyz.shape[0],4), dtype=self.get_xyz.dtype, requires_grad=False, device=self.device) 

        self.xys,  screenspace_points, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(means,  screenspace_points ,cholesky_elements, self.H, self.W, self.tile_bounds)
        out_img, per_pix_gs_nums,  screenspace_points= rasterize_gaussians_sum(self.xys, screenspace_points, depths, self.radii, conics, num_tiles_hit,
                colors, self._opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
        out_img = torch.clamp(out_img, 0, 1)
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        vq_loss = l_vqm + l_vqs + l_vqr + l_vqc
        return {"render": out_img, "vq_loss": vq_loss, "unit_bit":[m_bit, s_bit, r_bit, c_bit]}

    def train_iter_quantize(self, gt_image):
        render_pkg = self.forward_quantize()
        image = render_pkg["render"]
        loss = loss_fn(image, gt_image, self.loss_type, lambda_value=0.7) + render_pkg["vq_loss"]
        loss.backward()
        with torch.no_grad():
            mse_loss = F.mse_loss(image, gt_image)
            psnr = 10 * math.log10(1.0 / mse_loss.item())
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        if self.cholesky_optimizer is not None:
            self.cholesky_optimizer.zero_grad()
            self.cholesky_optimizer.step()
        self.scheduler.step()
        return loss, psnr

    def compress_wo_ec(self):
        means = torch.tanh(self.xyz_quantizer(self._xyz))
        # todo 改成直接量化压缩合并后的协方差矩阵
        cholesky_elements ,quant_cholesky_elements = self.cholesky_quantizer.compress(self._cholesky)
        cholesky_elements = cholesky_elements + self.cholesky_bound
        colors, feature_dc_index = self.features_dc_quantizer.compress(self.get_features)
        return {"xyz":self._xyz.half(), "feature_dc_index": feature_dc_index, "quant_cholesky_elements": quant_cholesky_elements}

    def decompress_wo_ec(self, encoding_dict):
        xyz, feature_dc_index, quant_cholesky_elements = encoding_dict["xyz"], encoding_dict["feature_dc_index"], encoding_dict["quant_cholesky_elements"]
        means = torch.tanh(xyz.float())
        cholesky_elements = self.cholesky_quantizer.decompress(quant_cholesky_elements)
        cholesky_elements = cholesky_elements + self.cholesky_bound
        colors = self.features_dc_quantizer.decompress(feature_dc_index)
        screenspace_points = torch.zeros((means.shape[0],4), dtype=self.get_xyz.dtype, requires_grad=False, device=self.device)

        self.xys, screenspace_points,depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(means, screenspace_points, cholesky_elements, self.H, self.W, self.tile_bounds)
        out_img, per_pix_gs_nums,  screenspace_points = rasterize_gaussians_sum(self.xys, screenspace_points, depths, self.radii, conics, num_tiles_hit,
                colors, self._opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
        out_img = torch.clamp(out_img, 0, 1)
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        return {"render":out_img}

    def analysis_wo_ec(self, encoding_dict):
        '''
            分析码率
        '''
        quant_cholesky_elements, feature_dc_index = encoding_dict["quant_cholesky_elements"], encoding_dict["feature_dc_index"]
        total_bits = 0
        initial_bits, codebook_bits = 0, 0
        for quantizer_index, layer in enumerate(self.features_dc_quantizer.quantizer.layers):
            codebook_bits += layer._codebook.embed.numel()*torch.finfo(layer._codebook.embed.dtype).bits
        initial_bits += self.cholesky_quantizer.scale.numel()*torch.finfo(self.cholesky_quantizer.scale.dtype).bits
        initial_bits += self.cholesky_quantizer.beta.numel()*torch.finfo(self.cholesky_quantizer.beta.dtype).bits
        initial_bits += codebook_bits

        total_bits += initial_bits
        total_bits += self._xyz.numel()*16

        feature_dc_index = feature_dc_index.int().cpu().numpy()
        index_max = np.max(feature_dc_index)
        max_bit = np.ceil(np.log2(index_max)) #calculate max bit for feature_dc_index
        total_bits += feature_dc_index.size * max_bit #get_np_size(encoding_dict["feature_dc_index"]) * 8
        
        quant_cholesky_elements = quant_cholesky_elements.cpu().numpy()
        total_bits += quant_cholesky_elements.size * 6 #cholesky bits 

        position_bits = self._xyz.numel()*16
        cholesky_bits, feature_dc_bits = 0, 0
        cholesky_bits += self.cholesky_quantizer.scale.numel()*torch.finfo(self.cholesky_quantizer.scale.dtype).bits
        cholesky_bits += self.cholesky_quantizer.beta.numel()*torch.finfo(self.cholesky_quantizer.beta.dtype).bits
        cholesky_bits += quant_cholesky_elements.size * 6
        feature_dc_bits += codebook_bits
        feature_dc_bits += feature_dc_index.size * max_bit

        bpp = total_bits/self.H/self.W
        position_bpp = position_bits/self.H/self.W
        cholesky_bpp = cholesky_bits/self.H/self.W
        feature_dc_bpp = feature_dc_bits/self.H/self.W
        return {"bpp": bpp, "position_bpp": position_bpp, 
            "cholesky_bpp": cholesky_bpp, "feature_dc_bpp": feature_dc_bpp}


    def compress(self):
        means = torch.tanh(self.xyz_quantizer(self._xyz))
        quant_cholesky_elements, cholesky_elements = self.cholesky_quantizer.compress(self._cholesky)
        cholesky_elements = cholesky_elements + self.cholesky_bound
        colors, feature_dc_index = self.features_dc_quantizer.compress(self.get_features)
        cholesky_compressed, cholesky_histogram_table, cholesky_unique = compress_matrix_flatten_categorical(quant_cholesky_elements.int().flatten().tolist())
        feature_dc_compressed, feature_dc_histogram_table, feature_dc_unique = compress_matrix_flatten_categorical(feature_dc_index.int().flatten().tolist())
        return {"xyz":self._xyz.half(), "feature_dc_index": feature_dc_index, "quant_cholesky_elements": quant_cholesky_elements, 
            "feature_dc_bitstream":[feature_dc_compressed, feature_dc_histogram_table, feature_dc_unique], 
            "cholesky_bitstream":[cholesky_compressed, cholesky_histogram_table, cholesky_unique]}

    def decompress(self, encoding_dict):
        xyz = encoding_dict["xyz"]
        num_points, device = xyz.size(0), xyz.device
        feature_dc_compressed, feature_dc_histogram_table, feature_dc_unique = encoding_dict["feature_dc_bitstream"]
        cholesky_compressed, cholesky_histogram_table, cholesky_unique = encoding_dict["cholesky_bitstream"]
        feature_dc_index = decompress_matrix_flatten_categorical(feature_dc_compressed, feature_dc_histogram_table, feature_dc_unique, num_points*2, (num_points, 2))
        quant_cholesky_elements = decompress_matrix_flatten_categorical(cholesky_compressed, cholesky_histogram_table, cholesky_unique, num_points*3, (num_points, 3))
        feature_dc_index = torch.from_numpy(feature_dc_index).to(device).int() #[800, 2]
        quant_cholesky_elements = torch.from_numpy(quant_cholesky_elements).to(device).float() #[800, 3]

        means = torch.tanh(xyz.float())
        cholesky_elements = self.cholesky_quantizer.decompress(quant_cholesky_elements)
        cholesky_elements = cholesky_elements + self.cholesky_bound
        colors = self.features_dc_quantizer.decompress(feature_dc_index)
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(means, cholesky_elements, self.H, self.W, self.tile_bounds)
        out_img = rasterize_gaussians_sum(self.xys, depths, self.radii, conics, num_tiles_hit,
                colors, self._opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
        out_img = torch.clamp(out_img, 0, 1)
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        return {"render":out_img}
   
    def analysis(self, encoding_dict):
        quant_cholesky_elements, feature_dc_index = encoding_dict["quant_cholesky_elements"], encoding_dict["feature_dc_index"]
        cholesky_compressed, cholesky_histogram_table, cholesky_unique = compress_matrix_flatten_categorical(quant_cholesky_elements.int().flatten().tolist())
        feature_dc_compressed, feature_dc_histogram_table, feature_dc_unique = compress_matrix_flatten_categorical(feature_dc_index.int().flatten().tolist())  
        cholesky_lookup = dict(zip(cholesky_unique, cholesky_histogram_table.astype(np.float64) / np.sum(cholesky_histogram_table).astype(np.float64)))
        feature_dc_lookup = dict(zip(feature_dc_unique, feature_dc_histogram_table.astype(np.float64) / np.sum(feature_dc_histogram_table).astype(np.float64)))

        total_bits = 0
        initial_bits, codebook_bits = 0, 0
        for quantizer_index, layer in enumerate(self.features_dc_quantizer.quantizer.layers):
            codebook_bits += layer._codebook.embed.numel()*torch.finfo(layer._codebook.embed.dtype).bits
        initial_bits += self.cholesky_quantizer.scale.numel()*torch.finfo(self.cholesky_quantizer.scale.dtype).bits
        initial_bits += self.cholesky_quantizer.beta.numel()*torch.finfo(self.cholesky_quantizer.beta.dtype).bits
        initial_bits += get_np_size(cholesky_histogram_table) * 8
        initial_bits += get_np_size(cholesky_unique) * 8 
        initial_bits += get_np_size(feature_dc_histogram_table) * 8
        initial_bits += get_np_size(feature_dc_unique) * 8  
        initial_bits += codebook_bits

        total_bits += initial_bits
        total_bits += self._xyz.numel()*16
        total_bits += get_np_size(cholesky_compressed) * 8
        total_bits += get_np_size(feature_dc_compressed) * 8

        position_bits = self._xyz.numel()*16
        cholesky_bits, feature_dc_bits = 0, 0
        cholesky_bits += self.cholesky_quantizer.scale.numel()*torch.finfo(self.cholesky_quantizer.scale.dtype).bits
        cholesky_bits += self.cholesky_quantizer.beta.numel()*torch.finfo(self.cholesky_quantizer.beta.dtype).bits
        cholesky_bits += get_np_size(cholesky_histogram_table) * 8
        cholesky_bits += get_np_size(cholesky_unique) * 8   
        cholesky_bits += get_np_size(cholesky_compressed) * 8
        feature_dc_bits += codebook_bits
        feature_dc_bits += get_np_size(feature_dc_histogram_table) * 8
        feature_dc_bits += get_np_size(feature_dc_unique) * 8  
        feature_dc_bits += get_np_size(feature_dc_compressed) * 8

        bpp = total_bits/self.H/self.W
        position_bpp = position_bits/self.H/self.W
        cholesky_bpp = cholesky_bits/self.H/self.W
        feature_dc_bpp = feature_dc_bits/self.H/self.W
        return {"bpp": bpp, "position_bpp": position_bpp, 
            "cholesky_bpp": cholesky_bpp, "feature_dc_bpp": feature_dc_bpp,}
