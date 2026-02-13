import math
import time
from pathlib import Path
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ms_ssim,ssim
import sys
from PIL import Image, ImageOps
import torch.nn.functional as F
from pytorch_msssim import ms_ssim, ssim
from utils import *
from tqdm import tqdm
import random
import torchvision.transforms as transforms
import wandb
import copy
import json
from models.utils import loss_fn
from models.gaussianimage_covariance import GaussianImage_Covariance



class SimpleTrainer2d:
    """Trains random 2d gaussians to fit an image."""

    def __init__(
            self,
            image_path: Path,
            log_dir: str,
            num_points: int = 2000,
            iterations: int = 30000,
            model_path=None,
            args=None,
    ):
        self.device = torch.device("cuda:0")
        gt_image = image_path_to_tensor(image_path)

        self.gt_image = gt_image.to(self.device)

        self.args = args
        self.num_points = num_points
        self.max_num_points = args.max_num_points
        image_path = Path(image_path)
        self.image_name = image_path.stem
        BLOCK_H, BLOCK_W = 16, 16
        self.H, self.W = self.gt_image.shape[2], self.gt_image.shape[3]
        self.iterations = iterations
        self.save_imgs = args.save_imgs
        self.loss_type = args.loss_type

        self.add_stage = 0
        self.log_dir = Path(os.path.join(log_dir, self.image_name))
        self.print = args.print
        self.resume = False
        self.logwriter = LogWriter(self.log_dir)
        checkpoint = {}
        if model_path is not None and os.path.exists(model_path):
            print(f"loading model path:{model_path}")
            self.logwriter.write(f"loading model path:{model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            self.num_points = checkpoint['num_gs']
            self.gaussian_model = GaussianImage_Covariance(loss_type=self.loss_type, opt_type=args.opt_type,
                                                       num_points=self.num_points, H=self.H, W=self.W,
                                                       BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
                                                       device=self.device, lr=args.lr, quantize=args.quantize,
                                                       args=args, logwriter=self.logwriter).to(self.device)
            model_dict = self.gaussian_model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint['gs'].items() if k in model_dict}
            self.gaussian_model.cholesky_bound = checkpoint['slv_bound']
            model_dict.update(pretrained_dict)
            self.gaussian_model.load_state_dict(model_dict)
            # self.gaussian_model.training_setup(args.lr,update_optimizer=True)
            self.resume = True
        else:
            self.gaussian_model = GaussianImage_Covariance(loss_type=self.loss_type, opt_type=args.opt_type,
                                                           num_points=self.num_points, H=self.H, W=self.W,
                                                           BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
                                                           device=self.device, lr=args.lr, quantize=args.quantize,
                                                           args=args, logwriter=self.logwriter).to(self.device)

    def add_sample_positions(self, render_image, iter=0):

        errors = torch.abs(render_image - self.gt_image).sum(dim=1)

        normalized_gradient = errors / torch.sum(errors)

        base_num_samples = 1000

        if iter == self.iterations - self.args.grow_iter:
            dynamic_num_samples = max(0, self.max_num_points - self.gaussian_model.cur_num_points)  # 最大采样点数量限制
        else:
            dynamic_num_samples = max(0, min(base_num_samples,
                                             self.max_num_points - self.gaussian_model.cur_num_points))
        if dynamic_num_samples:
            P_flat = normalized_gradient.view(-1)
            _, sampled_indices_1d = torch.topk(P_flat, dynamic_num_samples)

            # 将一维索引转换为二维坐标
            sampled_y = sampled_indices_1d // self.W
            sampled_x = sampled_indices_1d % self.W
            color = torch.zeros(dynamic_num_samples, 3, device=self.device)  #
            new_points_nums = sampled_y.shape[0]
            sampled_xyz = torch.stack([sampled_x, sampled_y], dim=1)
            new_attributes = {"new_xyz": sampled_xyz.float().to(self.device),
                              "new_features_dc": color.float(),
                              "new_cov2d": torch.rand(new_points_nums, 3).to(self.device) + torch.tensor(
                                  [0.5, 0, 0.5]).to(self.device)
                              }

            now_points_nums, non_definite = self.gaussian_model.densification_postfix(**new_attributes)
            self.logwriter.write(
                f"\n iter:{iter} , Add {dynamic_num_samples} new points But {non_definite} non-defite; cur {now_points_nums} points")
            return dynamic_num_samples
        return 0

    def train(self):
        psnr_list, iter_list = [], []
        progress_bar = tqdm(range(1, self.iterations + 1), desc="Training progress")
        best_psnr = 0

        self.gaussian_model.train()
        torch.cuda.synchronize()
        start_time = time.time()

        for iter in range(1, self.iterations + 1):
            loss, psnr, out_image = self.gaussian_model.train_iter(self.H, self.W, self.gt_image, isprint=self.print)
            psnr_list.append(psnr)
            iter_list.append(iter)
            with torch.no_grad():
                if best_psnr < psnr:
                    best_psnr = psnr
                    # if iter % 1000 == 0:
                    best_model_dict = copy.deepcopy(self.gaussian_model.state_dict())
                    slv_bound = copy.deepcopy(self.gaussian_model.cholesky_bound)
                    best_iter = iter

                if iter % 1000 == 0:
                    progress_bar.set_postfix(
                        {f"Loss": f"{loss.item():.{7}f}", "PSNR": f"{psnr:.{4}f}", "Best PSNR": f"{best_psnr:.{4}f}",
                         "best_iter": f"{best_iter}"})
                    progress_bar.update(1000)

                if iter % self.args.prune_iter == 0:
                    none_definite, cur_gs_nums = self.gaussian_model.non_semi_definite_prune(self.H, self.W)

                if self.args.adaptive_add:
                    if iter % self.args.grow_iter == 0 and iter < self.iterations:
                        new_added_points_nums = self.add_sample_positions(out_image, iter=iter)

        torch.cuda.synchronize()
        end_time = time.time() - start_time
        progress_bar.close()

        self.gaussian_model._xyz = nn.Parameter(best_model_dict['_xyz'])
        self.gaussian_model._features_dc = nn.Parameter(best_model_dict['_features_dc'])
        self.gaussian_model._cov2d = nn.Parameter(best_model_dict['_cov2d'])
        self.gaussian_model._opacity = nn.Parameter(best_model_dict['_opacity'])
        self.gaussian_model.load_state_dict(best_model_dict)
        self.gaussian_model.cholesky_bound = slv_bound
        self.gaussian_model.cur_num_points = best_model_dict['_xyz'].shape[0]

        # none_definite,cur_gs_nums=self.gaussian_model.non_semi_definite_prune(self.H,self.W)
        psnr_value, ms_ssim_value, test_end_time, FPS = self.test()

        total_params = sum([p.numel() for p in self.gaussian_model.parameters() if p.requires_grad]) / 1e6
        self.logwriter.write(
            "Training Complete in {:.4f}, Eval time:{:.8f}, FPS:{:.4f}, gs_nums:{:.2e},gs_params:{:.4f}M".format(
                end_time, test_end_time, FPS, self.gaussian_model.cur_num_points, total_params))
        torch.save({"gs": best_model_dict, "num_gs": self.gaussian_model.cur_num_points,
                    "psnr": best_psnr, "ms-ssim": ms_ssim_value, "slv_bound": slv_bound},
                   self.log_dir / "gaussian_model.pth.tar")
        return psnr_value, ms_ssim_value, end_time, test_end_time, FPS

    def test(self):
        self.gaussian_model.eval()
        with torch.no_grad():
            out = self.gaussian_model(H=self.H, W=self.W)
            torch.cuda.synchronize()
            test_start_time = time.time()
            for i in range(100):
                _ = self.gaussian_model(H=self.H, W=self.W)
            torch.cuda.synchronize()
            test_end_time = (time.time() - test_start_time) / 100
        mse_loss = F.mse_loss(out["render"].float(), self.gt_image.float())
        psnr = 10 * math.log10(1.0 / mse_loss.item())
        ms_ssim_value = ms_ssim(out["render"].float(), self.gt_image.float(), data_range=1, size_average=True).item()
        return (psnr, ms_ssim_value, test_end_time, 1 / test_end_time)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-d", "--dataset", type=str, default='./datasets/kodak/', help="Training dataset"
    )

    parser.add_argument(
        "--data_name", type=str, default='kodak', help="Training dataset"
    )
    parser.add_argument(
        "--iterations", type=int, default=50000, help="number of training epochs (default: %(default)s)"
    )
    parser.add_argument(
        "--prune_iter", type=int, default=100, help="iteration of each pruning  (default: %(default)s)"
    )
    parser.add_argument(
        "--grow_iter", type=int, default=5000, help="iteration of each growing (default: %(default)s)"
    )
    parser.add_argument(
        "--model_name", type=str, default="GaussianImage_Covariance",
        help="model selection: GaussianImage_Cholesky, GaussianImage_RS, 3DGS"
    )

    parser.add_argument(
        "--sh_degree", type=int, default=3, help="SH degree (default: %(default)s)"
    )
    parser.add_argument("--num_points", type=int, default=2500, help="2D GS points (default: %(default)s)")
    parser.add_argument("--max_num_points", type=int, default=5000, help="max 2D GS points (default: %(default)s)", )
    parser.add_argument("--opt_type", type=str, default="adam", help="the type of optimizer")
    parser.add_argument("-opt", "--opt_nums", type=int, default=1, help="the nums of optimizer")
    parser.add_argument("--model_path", type=str,  default=None, help="Path to a checkpoint")
    parser.add_argument("--seed", type=int, default=3047, help="Set random seed for reproducibility")
    parser.add_argument("--save_imgs", action="store_true", help="Save image")
    parser.add_argument("--print", type=bool, default=False, help="if need print details")
    parser.add_argument("--lr", type=float, default=0.018,help="Learning rate (default: %(default)s)")
    parser.add_argument("--warmup_iter", type=float, default=15000)
    parser.add_argument('--radius_clip', type=float, default=1.0)
    parser.add_argument("--prune", type=bool, default=True,  help="turn on pruning")
    parser.add_argument("--adaptive_add", type=bool, default=True, help="turn on adaptive add densification")
    parser.add_argument("--wandb-project", type=str, default=None, help='Weights & Biases Project')
    parser.add_argument("--loss_type", type=str, default="L2")
    parser.add_argument("--SLV_init", type=bool, default=True, help="if turn on CAF filter")
    parser.add_argument("--color_norm",action='store_true')
    parser.add_argument("--coords_norm", action='store_true', help="if normalize the coordinates")
    parser.add_argument("--coords_act", type=str, default="tanh", help="tanh")
    parser.add_argument("--save_interval", type=int, default=5, help="save interval")
    parser.add_argument("--clip_coe", type=float, default=3.)
    #  quantization parameters =======================================
    parser.add_argument("--quantize", type=bool, default=False, help="Quantize")
    parser.add_argument("--cov_quant", type=str, default="lsq", help="type of covariance quantization")
    parser.add_argument("--color_quant", type=str, default="lsq")
    parser.add_argument("--xy_quant", type=str, default="lsq")
    parser.add_argument("--xy_bit", type=int, default=12, help="bitdepth of xy attri")
    parser.add_argument("--cov_bit", type=int, default=10, help="bitdepth of cov attri")
    parser.add_argument("--color_bit", type=int, default=6, help="bitdepth of color attri")

    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    if args.model_name != "GaussianImage_Covariance":
        args.lr = 0.001
        args.opt_type = "adan"
        args.adaptive_add = False
        args.prune = False
        args.opacity = False
        args.opt_nums = 2

    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    log_dir = (
            f"./checkpoints/{args.data_name}/{args.model_name}_I{args.iterations}_N{args.num_points}{'_SLV' if args.SLV_init else ''}_R{args.radius_clip}{'_add' if args.adaptive_add else ''}" +
            f"{'_prune' if args.prune else ''}{'_colornorm' if args.color_norm else ''}"
    )
    logwriter = LogWriter(Path(log_dir))

    script_name = os.path.basename(__file__)
    logwriter.write(script_name)
    logwriter.write(args_text)

    psnrs, ms_ssims, training_times, eval_times, eval_fpses, gs_nums, params = [], [], [], [], [], [], []

    image_h, image_w = 0, 0
    image_length, start = 24, 0
    if args.data_name == "DIV2K_valid_HR":
        image_length, start = 100, 800

    model_path = (
            f"./checkpoints/{args.data_name}/{args.model_name}_I{args.iterations}_N{args.num_points}{'_SLV' if args.SLV_init else ''}_R{args.radius_clip}{'_add' if args.adaptive_add else ''}" +
            f"{'_prune' if args.prune else ''}{'_colornorm' if args.color_norm else ''}"
    )
    for i in range(start, start + image_length):
        image_path = Path(args.dataset) / f'kodim{i + 1:02}.png'
        model_path = Path(model_path) / f'kodim{i + 1:02}' / 'gaussian_model.pth.tar'

        if args.data_name == "DIV2K_valid_HR":
            image_path = Path(args.dataset) / f'{i + 1:04}.png'
            model_path = Path(args.model_path) / f'{i + 1:04}' / 'gaussian_model.pth.tar'

        trainer = SimpleTrainer2d(image_path=image_path, num_points=args.num_points,
                                  iterations=args.iterations,  args=args,
                                  model_path=model_path, log_dir=log_dir)

        #  ===========overfiting training=================

        psnr, ms_ssim_v, training_time, eval_time, eval_fps = trainer.train()
        psnrs.append(psnr)
        ms_ssims.append(ms_ssim_v)
        training_times.append(training_time)
        eval_times.append(eval_time)
        eval_fpses.append(eval_fps)
        image_h += trainer.H
        image_w += trainer.W
        image_name = image_path.stem
        finally_gs_nums = trainer.gaussian_model.cur_num_points
        finally_params = sum([p.numel() for p in trainer.gaussian_model.parameters() if p.requires_grad])
        gs_nums.append(finally_gs_nums)
        params.append(finally_params / 1e6)
        logwriter.write(
            "{}\t{}x{}\tPSNR\t{:.4f}\tMS-SSIM\t{:.4f}\tTraining\t{:.4f}\tEval\t{:.8f}\tFPS\t{:.4f}\tgs_nums\t{:.2e}\tParams(M)\t{:.2f}".format(
                image_name, trainer.H, trainer.W, psnr, ms_ssim_v, training_time,
                eval_time, eval_fps, finally_gs_nums, finally_params / 1e6))

    # representation recording===========
    avg_psnr = torch.tensor(psnrs).mean().item()
    avg_ms_ssim = torch.tensor(ms_ssims).mean().item()
    avg_training_time = torch.tensor(training_times).mean().item()
    avg_eval_time = torch.tensor(eval_times).mean().item()
    avg_eval_fps = torch.tensor(eval_fpses).mean().item()
    avg_h = image_h // image_length
    avg_w = image_w // image_length
    avg_gs_nums = sum(gs_nums) / image_length
    avg_params = sum(params) / image_length

    logwriter.write(
        "Average: {}x{}, PSNR:{:.4f}, MS-SSIM:{:.4f}, Training:{:.4f}s, Eval:{:.8f}s, FPS:{:.4f}, gs_nums:{:.2e}, Params(M):{:.2f}".format(
            avg_h, avg_w, avg_psnr, avg_ms_ssim, avg_training_time, avg_eval_time, avg_eval_fps, avg_gs_nums,
            avg_params))


if __name__ == "__main__":
    main(sys.argv[1:])
