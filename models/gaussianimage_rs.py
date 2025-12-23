from gsplat.project_gaussians_2d_scale_rot import project_gaussians_2d_scale_rot
from gsplat.rasterize_sum import rasterize_gaussians_sum
from pytorch_msssim import SSIM
from utils import *
import torch
import torch.nn as nn
import numpy as np
import math
from quantize import *
from optimizer import Adan

class GaussianImage_RS(nn.Module):
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
        self.lamda = kwargs["lamda"]
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
        pretrained_dict=kwargs['pretrained_dict']
        self.cov_quant=kwargs["args"].cov_quant
        self.color_quant=kwargs["args"].color_quant
        self.xy_quant=kwargs["args"].xy_quant
        self.distribution=kwargs["args"].distribution
        self.con_entropy_min=kwargs["args"].cem
        self.score_entropy=kwargs["args"].score_entropy
        self.prune_iterations=kwargs["args"].prune_iterations
        self.train_mask_iters=kwargs["args"].train_mask_iters
        self.parameters_decoder=kwargs["args"].latent_dec
        self.logwriter=kwargs['logwriter']
        self.rotation_activation = torch.sigmoid
        self.opacity_activation = torch.sigmoid if self.learnable_opacity else lambda x: x
        self.rgb_activation=torch.sigmoid if self.color_norm else lambda x: x
        self.bkcolor_activation=torch.sigmoid if self.bkcolor_norm else lambda x: x
        self.coords_activation=torch.tanh if self.coords_norm else lambda x: x
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.register_buffer('bound', torch.tensor([0.5, 0.5],device=self.device).view(1, 2))
        self.quantize = kwargs["quantize"]
        if pretrained_dict is not None:
            to_prune_nums,valid_points_mask=check_non_semi_definite(pretrained_dict["_cov2d"])
            self._xyz = nn.Parameter(pretrained_dict['_xyz'][valid_points_mask])
            self.init_num_points=self._xyz.shape[0]
            scaling_factors, rotation_angles=extract_scaling_rotation(pretrained_dict["_cov2d"][valid_points_mask])
            self._scaling = nn.Parameter(scaling_factors-self.bound)
            # self._scaling = nn.Parameter(scaling_factors)

            self._rotation = nn.Parameter(rotation_angles)
            self._features_dc = nn.Parameter(pretrained_dict['_features_dc'][valid_points_mask])
        else:
            self._xyz = nn.Parameter(torch.atanh(2 * (torch.rand(self.init_num_points, 2) - 0.5)))
            self._scaling = nn.Parameter(torch.rand(self.init_num_points, 2))
            self._rotation = nn.Parameter(torch.rand(self.init_num_points, 1))
            self._features_dc = nn.Parameter(torch.rand(self.init_num_points, 3))
        if self.learnable_opacity  :
            # self._opacity= nn.Parameter(torch.logit(0.1 * torch.ones(self.init_num_points, 1)))
            self._opacity= nn.Parameter(torch.rand(self.init_num_points, 1))
        else:
            self.register_buffer('_opacity', torch.ones((self.init_num_points, 1)))
        

        self.last_size = (self.H, self.W)
        if self.learnable_back_color:
            self.background =  nn.Parameter(torch.zeros(self.init_num_points, 3, device=self.device))
        else:
            self.register_buffer('background', torch.ones(3))
        
        # self.bound=torch.tensor([self.H*self.W/(9*torch.pi*self.init_num_points),
                                        #   self.H*self.W/(9*torch.pi*self.init_num_points)]).view(1, 2).to(self.device)
        #  # densification
        # self.max_radii2D = torch.zeros(self.init_num_points,device=self.device)
        # self.xyz_gradient_accum = torch.zeros((self.init_num_points, 1), device=self.device)
        # self.denom = torch.zeros((self.init_num_points, 1), device=self.device)
      
        if self.quantize:
            self.xyz_quantizer = FakeQuantizationHalf.apply 
            self.features_dc_quantizer = VectorQuantizer(codebook_dim=3, codebook_size=8, num_quantizers=2, vector_type="vector", kmeans_iters=5) 
            self.scaling_quantizer = UniformQuantizer(signed=False, bits=6, learned=True, num_channels=2) 
            self.rotation_quantizer = UniformQuantizer(signed=False, bits=6, learned=True, num_channels=1)

        if kwargs["opt_type"] == "adam":
            l = [
                {'params': [self._xyz], 'lr': kwargs["lr"], "name": "xyz"},
                {'params': [self._features_dc], 'lr': kwargs["lr"], "name": "f_dc"},
                # {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
                # {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': kwargs["lr"], "name": "scaling"},
                {'params': [self._rotation], 'lr': kwargs["lr"], "name": "rotation"}
            ]

            self.optimizer = torch.optim.Adam(l, lr=kwargs["lr"])
        else:
            self.optimizer = Adan(self.parameters(), lr=kwargs["lr"])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20000, gamma=0.5)

    def training_setup(self,lr,update_optimizer=False,quantize=False,iter=0):
        if update_optimizer:
            l=[
                    {'params': [self._xyz], 'lr':lr, "name": "xyz"},
                    {'params': [self._features_dc], 'lr': lr, "name": "f_dc"},
                      {'params': [self._scaling], 'lr': lr, "name": "scaling"},
                    {'params': [self._rotation], 'lr': lr, "name": "rotation"}
                ]
                
            if self.learnable_opacity:
                l.append({'params': [self._opacity], 'lr': lr, "name": "opacity"})
            if self.learnable_back_color:
                l.append({'params': [self.background], 'lr': lr, "name": "bkcolor"})
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        if quantize:
            self.quantize=True
            if self.xy_quant=='lsq':
                self.xyz_quantizer= UniformQuantizer(signed=False, bits=12, learned=True, num_channels=2).to(self.device)  #better than fp16
                self.xyz_quantizer_optimizer= torch.optim.Adam(self.xyz_quantizer.parameters(), lr=lr)
                self.xyz_scheduler=torch.optim.lr_scheduler.StepLR(self.xyz_quantizer_optimizer , step_size=10000, gamma=0.5)
            else:
                self.xyz_quantizer=FakeQuantizationHalf.apply
            if self.cov_quant=='log':
                self.cholesky_quantizer =LogQuantizer(signed=False, bits=10,learned=False, num_channels=3)

            elif self.cov_quant=='lsq':
                
                self.scaling_quantizer = UniformQuantizer(signed=False, bits=6, learned=True, num_channels=2) 
                self.rotation_quantizer = UniformQuantizer(signed=True, bits=6, learned=True, num_channels=1) 
                # 将两个量化器的参数结合在一起
                all_params = list(self.scaling_quantizer.parameters()) + list(self.rotation_quantizer.parameters())
                self.cov2d_quantizer_optimizer = torch.optim.Adam(all_params, lr=lr, eps=1e-15)
                self.cov2d_scheduler= torch.optim.lr_scheduler.StepLR(self.cov2d_quantizer_optimizer , step_size=10000, gamma=0.5)

            else:
                self.cholesky_quantizer=VectorQuantizer(codebook_dim=3, codebook_size=128, 
                            num_quantizers=3, vector_type="vector", kmeans_iters=5).to(self.device) 
            if self.color_quant=='vq':
                self.features_dc_quantizer = VectorQuantizer(codebook_dim=3, codebook_size=8, num_quantizers=2, vector_type="vector", kmeans_iters=5).to(self.device) 
            elif self.color_quant=='lsq':
                self.features_dc_quantizer = UniformQuantizer(signed=False, bits=6, learned=True, num_channels=3).to(self.device) 
                self.color_quantizer_optimizer = torch.optim.Adam(self.features_dc_quantizer.parameters(),lr=lr, eps=1e-15)
                self.color_scheduler= torch.optim.lr_scheduler.StepLR(self.color_quantizer_optimizer , step_size=10000, gamma=0.5)
            elif self.color_quant=='log':
                self.features_dc_quantizer =LogQuantizer(signed=False, bits=6, learned=False, num_channels=3)
                # self.features_dc_quantizer=LSQPlusActivationQuantizer(bits=6, all_positive=True, num_channels=3)

    def _init_data(self):
        self.scaling_quantizer._init_data(self._scaling)
        self.rotation_quantizer._init_data(self.get_rotation)

    @property
    def get_scaling(self):
        return torch.abs(self._scaling+self.bound)
        # return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)*2*math.pi # 0-2pi
        # return self._rotation
    
    
    @property
    def get_xyz(self):
        return self.coords_activation(self._xyz)
    
    @property
    def get_features(self):
        return self.rgb_activation(self._features_dc)
    
    @property
    def get_opacity(self):
        return self._opacity 
    def build_scaling_rotation(self, scaling, rotation):
        # 构建缩放旋转矩阵
        cos_theta = torch.cos(rotation)
        sin_theta = torch.sin(rotation)
        L = torch.zeros((scaling.shape[0], 2, 2))
        L[:, 0, 0] = scaling[:, 0] * cos_theta.squeeze()
        L[:, 0, 1] = -scaling[:, 1] * sin_theta.squeeze()
        L[:, 1, 0] = scaling[:, 0] * sin_theta.squeeze()
        L[:, 1, 1] = scaling[:, 1] * cos_theta.squeeze()
        return L
    def build_rotation(self,  rotation):
        # 构建缩放旋转矩阵
        cos_theta = torch.cos(rotation)
        sin_theta = torch.sin(rotation)
        R = torch.zeros((rotation.shape[0], 2, 2))
        R[:, 0, 0] =  cos_theta.squeeze()
        R[:, 0, 1] = - sin_theta.squeeze()
        R[:, 1, 0] = sin_theta.squeeze()
        R[:, 1, 1] =  cos_theta.squeeze()
        return R

    def strip_symmetric(self, matrix):
        # 提取对称部分
        return torch.stack([matrix[...,0,0],matrix[...,0,1],matrix[...,1,1]],dim=1)

    def build_covariance_from_scaling_rotation(self):
        scaling = self.get_scaling.detach().clone()
        rotation = self.get_rotation.detach().clone()
        L = self.build_scaling_rotation(scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        symm = self.strip_symmetric(actual_covariance)
        return symm
    def get_attributes(self):
        coords=self.xys.detach().clone().cpu().numpy()
        covs=self.build_covariance_from_scaling_rotation().detach().clone().cpu().numpy()
        colors =  self.get_features.detach().clone().cpu().numpy()
        return {'coords': coords, 'covs': covs, 'colors': colors}
    def forward(self):
        screenspace_points = torch.zeros((self.get_xyz.shape[0],4), dtype=self.get_xyz.dtype, requires_grad=True, device=self.device) + 0
        self.xys,screenspace_points, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d_scale_rot(self.get_xyz,screenspace_points,
                 self.get_scaling,self.get_rotation, self.H, self.W, self.tile_bounds,coords_norm=self.coords_norm )
        out_img ,per_pix_gs_nums,screenspace_points= rasterize_gaussians_sum(self.xys,screenspace_points, depths, self.radii, conics, num_tiles_hit,
                self.get_features, self.get_opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
        out_img = torch.clamp(out_img, 0, 1) #[H, W, 3]
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        return {"render": (out_img),
                 "num_tiles_hit":num_tiles_hit,
                "radiii":self.radii,
                "per_pix_gs_nums":per_pix_gs_nums,
                  "visibility_filter" : self.radii > 0}
    def render(self,H,W,mirror=False,isprint=False):
        screenspace_points = torch.zeros((self.get_xyz.shape[0],4), dtype=self.get_xyz.dtype, requires_grad=True, device=self.device) + 0
        tile_bounds = (
            (W + self.BLOCK_W - 1) // self.BLOCK_W,
            (H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        ) #
        self.xys,screenspace_points, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d_scale_rot(self.get_xyz, self.get_scaling,screenspace_points, 
                                        self.get_rotation, H, W, tile_bounds)
        out_img ,per_pix_gs_nums,screenspace_points= rasterize_gaussians_sum(self.xys,screenspace_points, depths, self.radii, conics, num_tiles_hit,
                self.get_features, self.get_opacity,H, W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
        out_img = torch.clamp(out_img, 0, 1) #[H, W, 3]
        out_img = out_img.view(-1, H, W, 3).permute(0, 3, 1, 2).contiguous()
      
        return {"render": (out_img), "num_tiles_hit":num_tiles_hit,
                "radiii":self.radii,
                "per_pix_gs_nums":per_pix_gs_nums,
                  "visibility_filter" : self.radii > 0,
                "screen_points":screenspace_points}
    def train_iter(self, gt_image,gt_mirror_image=None,isprint=False,iter=1):
        render_pkg = self.forward()
        image = render_pkg["render"]

        loss = loss_fn(image, gt_image, self.loss_type, lambda_value=0.7)
        loss.backward()
        with torch.no_grad():
            mse_loss = F.mse_loss(image, gt_image)
            psnr = 10 * math.log10(1.0 / mse_loss.item())

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none = True)
        self.scheduler.step()
        return loss, psnr,image.detach()
    def add_densification_stats(self, viewspace_point_tensor, update_filter):
            # self.xyz_gradient_accum[update_filter] += torch.norm(torch.abs(viewspace_point_tensor.grad[update_filter,:2]), dim=-1, keepdim=True)
            self.xyz_gradient_accum[update_filter] += torch.norm((viewspace_point_tensor.grad[update_filter,:2]), dim=-1, keepdim=True)
            self.denom[update_filter] += 1
    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc,new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        # "f_rest": new_features_rest,
        # "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}


        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        now_num_points=self._xyz.shape[0]
        
        self.register_buffer('_opacity' , torch.ones((now_num_points, 1), device=self.device))
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        
        self.xyz_gradient_accum = torch.zeros((now_num_points, 1), device=self.device) #累计梯度重新置为0
        self.denom = torch.zeros((now_num_points, 1), device=self.device)
        self.max_radii2D = torch.zeros((now_num_points),  device=self.device)
        return now_num_points
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
    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        new_num_points=self._xyz.shape[0]
        self.register_buffer('_opacity' , torch.ones((new_num_points, 1), device=self.device))
        # self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.init_num_points=new_num_points
        self.bound=torch.tensor([self.H*self.W/(9*torch.pi*self.init_num_points),
                                          self.H*self.W/(9*torch.pi*self.init_num_points)]).view(1, 2).to(self.device)
    
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        return torch.sum(valid_points_mask).item()
    def densify_and_split(self, grads, grad_threshold, scene_extent=200, N=2, abe_split=False):
        '''
            split into N=2 small gs 
        '''
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        if abe_split:
            BACK_N  = N - 1 
            padded_grad = torch.zeros((n_init_points),  device=self.device)
            padded_grad[:grads.shape[0]] = grads.squeeze()
            selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
            selected_pts_mask = torch.logical_and(selected_pts_mask,
                                                torch.max(self.get_scaling, dim=1).values > 0.01*scene_extent)
            selected_nums= torch.sum(selected_pts_mask).item()
            if selected_nums== 0: return 0,n_init_points
            stds = self.get_scaling[selected_pts_mask].repeat(BACK_N,1)
            means =torch.zeros((stds.size(0), 2), device=self.device)
            samples = torch.normal(mean=means, std=stds)
            rots = self.build_rotation(self._rotation[selected_pts_mask]).repeat(BACK_N,1,1)
            new_xyz = self.get_xyz[selected_pts_mask].repeat(BACK_N, 1)
            new_scaling = torch.log(self.get_scaling[selected_pts_mask].repeat(BACK_N,1))
            new_rotation = self._rotation[selected_pts_mask].repeat(BACK_N,1)
            new_features_dc = self._features_dc[selected_pts_mask].repeat(BACK_N,1,1)
            new_xyz = new_xyz*0.3*scene_extent
            
            self.densification_postfix(new_xyz, new_features_dc, new_scaling, new_rotation)
            n_init_points = self.get_xyz.shape[0]
        
        padded_grad = torch.zeros((n_init_points),  device=self.device)
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values >0.01*scene_extent)
        selected_nums= torch.sum(selected_pts_mask).item()
        if selected_nums== 0: return 0,n_init_points
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 2), device=self.device)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = torch.log(self.get_scaling[selected_pts_mask].repeat(N,1) / (self.divide_ratio*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        self.densification_postfix(new_xyz, new_features_dc, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device=self.device, dtype=bool)))
        new_num_points_pruned=self.prune_points(prune_filter)
        return selected_nums, new_num_points_pruned

    def densify_and_clone(self, grads, grad_threshold, scene_extent=1.1):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        # selected_pts_mask = torch.logical_and(selected_pts_mask,
        #                                      self.max_radii2D<= scene_extent)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= 0.01*scene_extent)
        selected_nums_clone=torch.sum(selected_pts_mask).item()
        if selected_nums_clone:
            new_xyz = self._xyz[selected_pts_mask]
            new_features_dc = self._features_dc[selected_pts_mask]
            # new_opacities = self.get_opacity[selected_pts_mask]
            new_scaling = self._scaling[selected_pts_mask]
            new_rotation = self._rotation[selected_pts_mask]
            new_num_points=self.densification_postfix(new_xyz, new_features_dc,new_scaling, new_rotation)
            return selected_nums_clone,new_num_points
        else:
            return 0,0
    def densify_and_prune(self, max_grad,  extent, max_screen_size,N=2, abe_split=False):#ltt
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        selected_nums_for_clone,new_num_points_cloned=self.densify_and_clone(grads, max_grad, extent)
        selected_nums_for_split, new_num_points_pruned=self.densify_and_split(grads, max_grad, extent,
                                                                              N=N, abe_split=abe_split)
        valid_pts_nums=new_num_points_pruned
        # prune_mask = (self.get_opacity < min_opacity).squeeze() # opacity小于阈值则剪枝
        # if max_screen_size:# 球半径 太大的要剪枝
        #     big_points_vs = self.max_radii2D > max_screen_size
        #     big_points_ws = self.get_scaling.max(dim=1).values> 0.1 * extent
        #     # prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        #     prune_mask = torch.logical_or(big_points_vs, big_points_ws)
        #     prune_mask = big_points_vs
        #     if torch.sum(prune_mask)>0:
        #         valid_pts_nums=self.prune_points(prune_mask) # 最后点数
        # else:
        #     valid_pts_nums=new_num_points_pruned

        torch.cuda.empty_cache()
        return valid_pts_nums,selected_nums_for_clone, selected_nums_for_split
    def forward_quantize(self):
        screenspace_points = torch.zeros((self.cur_num_points,4), dtype=torch.float32, requires_grad=True, device=self.device) + 0
        if self.xy_quant=='fp16':
            means =self.coords_activation(self.xyz_quantizer(self._xyz)) #fp16 28.1148
            l_vqm, m_bit = 0, 16*self.cur_num_points*2
        else:
            means, l_vqm ,m_bit ,code_xy= self.xyz_quantizer(self._xyz)  # better

        scaling, l_vqs, s_bit ,s_code= self.scaling_quantizer(self._scaling)
        # scaling = torch.abs(scaling + self.bound)
        # scaling=self.scaling_activation(scaling)
        # scaling, l_vqs, s_bit ,s_code= self.scaling_quantizer(self.get_scaling)

        rotation, l_vqr, r_bit,r_code = self.rotation_quantizer(self.get_rotation)

        if self.color_quant=='vq':
            colors, l_vqc, c_bit = self.features_dc_quantizer(self.get_features)
        else:
            colors, l_vqc, c_bit, code_color = self.features_dc_quantizer(self.get_features,quant_loss=self.con_entropy_min)  # better if directly quantize the color after activation fun
        
        self.xys, screenspace_points,depths, self.radii, conics, num_tiles_hit = project_gaussians_2d_scale_rot(means, screenspace_points,
                            scaling, rotation, self.H, self.W,
                             self.tile_bounds,coords_norm=self.coords_norm)
        out_img,per_pix_gs_nums,screenspace_points= rasterize_gaussians_sum(self.xys, screenspace_points,depths, self.radii, conics, num_tiles_hit,
                colors, self._opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
        out_img = torch.clamp(out_img, 0, 1)
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        vq_loss = l_vqm + l_vqs + l_vqr + l_vqc 
        return {"render": out_img, "vq_loss": vq_loss, "unit_bit":[m_bit, s_bit, r_bit, c_bit]}

    def train_iter_quantize(self, gt_image,iter=1):
        render_pkg = self.forward_quantize()
        image = render_pkg["render"]
        image_loss=loss_fn(image, gt_image, self.loss_type, lambda_value=0.7)
        loss = image_loss+ render_pkg["vq_loss"]
        loss.backward()
        with torch.no_grad():
            mse_loss = F.mse_loss(image, gt_image)
            psnr = 10 * math.log10(1.0 / mse_loss.item())
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none = True)
        self.scheduler.step()
        return image.detach(), loss.item(), image_loss.item(),render_pkg["vq_loss"],psnr

    def compress_wo_ec(self):
        if self.xy_quant=='fp16':
            means =self.xyz_quantizer(self._xyz) #fp16 28.1148
        else:
            means, quant_means= self.xyz_quantizer.compress(self._xyz)
        # means = self.coords_activation(means)
        quant_scaling, _ = self.scaling_quantizer.compress(self._scaling)
        quant_rotation, _ = self.rotation_quantizer.compress(self.get_rotation)
        _, feature_dc_index = self.features_dc_quantizer.compress(self.get_features)
        return {"xyz":means, "feature_dc_index": feature_dc_index, "quant_scaling": quant_scaling, "quant_rotation": quant_rotation}

    def decompress_wo_ec(self, encoding_dict):
        screenspace_points = torch.zeros((self.cur_num_points, 4), dtype=torch.float32, requires_grad=True, device=self.device) + 0
        xyz, quant_scaling, quant_rotation = encoding_dict["xyz"], encoding_dict["quant_scaling"], encoding_dict["quant_rotation"]
        feature_dc_index = encoding_dict["feature_dc_index"]
        means = self.coords_activation(xyz)
        scaling = self.scaling_quantizer.decompress(quant_scaling)
        scaling = torch.abs(scaling + self.bound)
        rotation = self.rotation_quantizer.decompress(quant_rotation)
        colors = self.features_dc_quantizer.decompress(feature_dc_index)
        self.xys, screenspace_points,depths, self.radii, conics, num_tiles_hit = project_gaussians_2d_scale_rot(means, screenspace_points,scaling, rotation, self.H, self.W, self.tile_bounds,
                coords_norm=self.coords_norm)
        out_img, _, _ = rasterize_gaussians_sum(self.xys,screenspace_points, depths, self.radii, conics, num_tiles_hit,
                colors, self._opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
        out_img = torch.clamp(out_img, 0, 1)
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        return {"render":out_img}
    
    def analysis_wo_ec(self, encoding_dict):
        quant_scaling, quant_rotation, feature_dc_index = encoding_dict["quant_scaling"], encoding_dict["quant_rotation"], encoding_dict["feature_dc_index"]

        total_bits = 0
        initial_bits, scaling_codebook_bits, rotation_codebook_bits, feature_dc_codebook_bits = 0, 0, 0, 0

        
        scaling_codebook_bits += self.scaling_quantizer.scale.numel()*torch.finfo(self.scaling_quantizer.scale.dtype).bits
        scaling_codebook_bits += self.scaling_quantizer.beta.numel()*torch.finfo(self.scaling_quantizer.beta.dtype).bits
        rotation_codebook_bits += self.rotation_quantizer.scale.numel()*torch.finfo(self.rotation_quantizer.scale.dtype).bits
        rotation_codebook_bits += self.rotation_quantizer.beta.numel()*torch.finfo(self.rotation_quantizer.beta.dtype).bits  

        for quantizer_index, layer in enumerate(self.features_dc_quantizer.quantizer.layers):
            feature_dc_codebook_bits += layer._codebook.embed.numel()*torch.finfo(layer._codebook.embed.dtype).bits

        initial_bits += scaling_codebook_bits
        initial_bits += rotation_codebook_bits
        initial_bits += feature_dc_codebook_bits

        quant_scaling, quant_rotation, feature_dc_index = quant_scaling.cpu().numpy(), quant_rotation.cpu().numpy(), feature_dc_index.cpu().numpy()
        total_bits += initial_bits
        total_bits += self._xyz.numel()*16 if self.xy_quant == "fp16" else self._xyz.numel()*12
        total_bits += quant_scaling.size * 6
        total_bits += quant_rotation.size * 6
        index_max = np.max(feature_dc_index)
        max_bit = np.ceil(np.log2(index_max))
        total_bits += feature_dc_index.size * max_bit

        position_bits = self._xyz.numel()*16 if self.xy_quant == "fp16" else self._xyz.numel()*12
        scaling_bits, rotation_bits, feature_dc_bits = 0, 0, 0
        scaling_bits += scaling_codebook_bits
        scaling_bits += quant_scaling.size * 6
        rotation_bits += rotation_codebook_bits
        rotation_bits += quant_rotation.size * 6
        feature_dc_bits += feature_dc_codebook_bits
        feature_dc_bits += feature_dc_index.size * max_bit

        bpp = total_bits/self.H/self.W
        position_bpp = position_bits/self.H/self.W
        scaling_bpp = scaling_bits/self.H/self.W
        rotation_bpp = rotation_bits/self.H/self.W
        feature_dc_bpp = feature_dc_bits/self.H/self.W
        cholesky_bpp = scaling_bpp+rotation_bpp
        return {"bpp": bpp, "position_bpp": position_bpp, 
            "cholesky_bpp": cholesky_bpp, "feature_dc_bpp": feature_dc_bpp, "scaling_bpp": scaling_bpp,
            "rotation_bpp": rotation_bpp}
    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda() # 10w,3
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 2)
        rots = torch.ones((fused_point_cloud.shape[0], 4), device="cuda")
        #rots[:, 1] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
    def compress(self):
        means = torch.tanh(self.xyz_quantizer(self._xyz))
        _, scaling_index = self.scaling_quantizer.compress(self._scaling)
        _, rotation_index = self.rotation_quantizer.compress(self.get_rotation)
        _, feature_dc_index = self.features_dc_quantizer.compress(self.get_features)
        return {"xyz":self._xyz.half(), "feature_dc_index": feature_dc_index, "scaling_index": scaling_index, "rotation_index": rotation_index}

    def decompress(self, encoding_dict):
        xyz, scaling_index, feature_dc_index = encoding_dict["xyz"], encoding_dict["scaling_index"], encoding_dict["feature_dc_index"]
        rotation_index = encoding_dict["rotation_index"]
        means = torch.tanh(xyz.float())
        scaling = self.scaling_quantizer.decompress(scaling_index)
        scaling = torch.abs(scaling + self.bound)
        rotation = self.rotation_quantizer.decompress(rotation_index)
        colors = self.features_dc_quantizer.decompress(feature_dc_index)
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d_scale_rot(means, scaling, rotation, self.H, self.W, self.tile_bounds)
        out_img = rasterize_gaussians_sum(self.xys, depths, self.radii, conics, num_tiles_hit,
                colors, self._opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
        out_img = torch.clamp(out_img, 0, 1)
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        return {"render":out_img}

    def analysis(self, encoding_dict):
        scaling_index, rotation_index, feature_dc_index = encoding_dict["scaling_index"], encoding_dict["rotation_index"], encoding_dict["feature_dc_index"]
        scaling_compressed, scaling_histogram_table, scaling_unique = compress_matrix_flatten_categorical(scaling_index.int().flatten().tolist())
        rotation_compressed, rotation_histogram_table, rotation_unique = compress_matrix_flatten_categorical(rotation_index.int().flatten().tolist())
        feature_dc_compressed, feature_dc_histogram_table, feature_dc_unique = compress_matrix_flatten_categorical(feature_dc_index.int().flatten().tolist())  
        scaling_lookup = dict(zip(scaling_unique, scaling_histogram_table.astype(np.float64) / np.sum(scaling_histogram_table).astype(np.float64)))
        rotation_lookup = dict(zip(rotation_unique, rotation_histogram_table.astype(np.float64) / np.sum(rotation_histogram_table).astype(np.float64)))
        feature_dc_lookup = dict(zip(feature_dc_unique, feature_dc_histogram_table.astype(np.float64) / np.sum(feature_dc_histogram_table).astype(np.float64)))

        total_bits = 0
        initial_bits, scaling_codebook_bits, rotation_codebook_bits, feature_dc_codebook_bits = 0, 0, 0, 0
        for quantizer_index, layer in enumerate(self.scaling_quantizer.quantizer.layers):
            scaling_codebook_bits += layer._codebook.embed.numel()*torch.finfo(layer._codebook.embed.dtype).bits
        for quantizer_index, layer in enumerate(self.rotation_quantizer.quantizer.layers):
            rotation_codebook_bits += layer._codebook.embed.numel()*torch.finfo(layer._codebook.embed.dtype).bits
        for quantizer_index, layer in enumerate(self.features_dc_quantizer.quantizer.layers):
            feature_dc_codebook_bits += layer._codebook.embed.numel()*torch.finfo(layer._codebook.embed.dtype).bits

        initial_bits += scaling_codebook_bits
        initial_bits += rotation_codebook_bits
        initial_bits += feature_dc_codebook_bits
        initial_bits += get_np_size(scaling_histogram_table) * 8
        initial_bits += get_np_size(scaling_unique) * 8 
        initial_bits += get_np_size(rotation_histogram_table) * 8
        initial_bits += get_np_size(rotation_unique) * 8 
        initial_bits += get_np_size(feature_dc_histogram_table) * 8
        initial_bits += get_np_size(feature_dc_unique) * 8  

        total_bits += initial_bits
        total_bits += self._xyz.numel()*16
        total_bits += get_np_size(scaling_compressed) * 8
        total_bits += get_np_size(rotation_compressed) * 8
        total_bits += get_np_size(feature_dc_compressed) * 8

        position_bits = self._xyz.numel()*16
        scaling_bits, rotation_bits, feature_dc_bits = 0, 0, 0
        scaling_bits += scaling_codebook_bits
        scaling_bits += get_np_size(scaling_histogram_table) * 8
        scaling_bits += get_np_size(scaling_unique) * 8   
        scaling_bits += get_np_size(scaling_compressed) * 8
        rotation_bits += rotation_codebook_bits
        rotation_bits += get_np_size(rotation_histogram_table) * 8
        rotation_bits += get_np_size(rotation_unique) * 8   
        rotation_bits += get_np_size(rotation_compressed) * 8
        feature_dc_bits += feature_dc_codebook_bits
        feature_dc_bits += get_np_size(feature_dc_histogram_table) * 8
        feature_dc_bits += get_np_size(feature_dc_unique) * 8  
        feature_dc_bits += get_np_size(feature_dc_compressed) * 8

        bpp = total_bits/self.H/self.W
        position_bpp = position_bits/self.H/self.W
        scaling_bpp = scaling_bits/self.H/self.W
        rotation_bpp = rotation_bits/self.H/self.W
        feature_dc_bpp = feature_dc_bits/self.H/self.W
        cholesky_bpp = scaling_bpp+rotation_bpp
        return {"bpp": bpp, "position_bpp": position_bpp, 
            "cholesky_bpp": cholesky_bpp, "feature_dc_bpp": feature_dc_bpp, "scaling_bpp": scaling_bpp,
            "rotation_bpp": rotation_bpp}

