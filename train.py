import math
import time
from pathlib import Path
import argparse
import yaml
import sys
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ms_ssim,ssim
from utils import *
from tqdm import tqdm
import random
import copy
import torchvision.transforms as transforms
from utils import image_path_to_tensor
from models.gaussianimage_covariance import GaussianImage_Covariance
# from models.gaussianimage_cholesky import GaussianImage_Cholesky
# from models.gaussianimage_rs import GaussianImage_RS


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
        self.log_dir = log_dir
        self.logwriter = LogWriter(self.log_dir)
        self.save_imgs = args.save_imgs
        self.using_wandb = args.wandb_project is not None
        self.loss_type = args.loss_type
        self.image_idx = 1
        self.add_stage = 0
        self.resume = False
        if model_path is not None and os.path.exists(model_path):
            print(f"loading model path:{model_path}")
            self.logwriter.write(f"loading model path:{model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            self.num_points = checkpoint['num_gs']
            self.gaussian_model = GaussianImage_Covariance(loss_type=self.loss_type, opt_type=args.opt_type,
                                                       num_points=self.num_points, H=self.H, W=self.W,
                                                       BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
                                                       device=self.device, lr=args.lr, quantize=args.quantize,
                                                       args=args, logwriter=self.logwriter,
                                                       ).to(self.device)
            model_dict = self.gaussian_model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint['gs'].items() if k in model_dict}
            self.gaussian_model.cholesky_bound = checkpoint['slv_bound']
            model_dict.update(pretrained_dict)
            self.gaussian_model.load_state_dict(model_dict)
            self.resume = True
        else:
            self.gaussian_model = GaussianImage_Covariance(loss_type=self.loss_type, opt_type=args.opt_type,
                                                           num_points=self.num_points, H=self.H, W=self.W,
                                                           BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
                                                           device=self.device, lr=args.lr, quantize=args.quantize,
                                                           args=args, logwriter=self.logwriter,
                                                           ).to(self.device)



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

            sampled_y = sampled_indices_1d // self.W
            sampled_x = sampled_indices_1d % self.W

            color = torch.zeros(dynamic_num_samples, 3, device=self.device)
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
        best_iter = 0
        identity = None
        self.gaussian_model.train()
        torch.cuda.synchronize()
        start_time = time.time()

        for iter in range(1, self.iterations):
            if iter < self.args.warmup_iter:
                loss, psnr, out_image = self.gaussian_model.train_iter(self.H, self.W, self.gt_image)
                img_loss = loss
                vq_loss = 0

            elif iter == self.args.warmup_iter:
                best_psnr = 0
                print("warmup finished! start training quantize")
                self.gaussian_model._xyz = nn.Parameter(best_model_dict['_xyz'])
                self.gaussian_model._features_dc = nn.Parameter(best_model_dict['_features_dc'])
                self.gaussian_model._cov2d = nn.Parameter(best_model_dict['_cov2d'])
                self.gaussian_model._opacity = nn.Parameter(best_model_dict['_opacity'])
                self.gaussian_model.load_state_dict(best_model_dict)
                self.gaussian_model.cholesky_bound = slv_bound
                self.gaussian_model.cur_num_points = best_model_dict['_xyz'].shape[0]
                lr = self.gaussian_model.scheduler.get_last_lr()[0]
                self.gaussian_model.training_setup(lr, update_optimizer=True, quantize=True)
                out_image, loss, img_loss, vq_loss, psnr = self.gaussian_model.train_iter_quantize(self.gt_image)
            else:
                out_image, loss, img_loss, vq_loss, psnr = self.gaussian_model.train_iter_quantize(self.gt_image)

            psnr_list.append(psnr)
            iter_list.append(iter)

            with torch.no_grad():
                if iter % 100 == 0:
                    progress_bar.set_postfix(
                        {f"Loss": f"{loss.item():.{6}f}", "PSNR": f"{psnr:.{4}f}", "Best PSNR": f"{best_psnr:.{4}f}",
                         "img_loss": f"{img_loss:.{6}f}",
                         # "VQ Loss": f"{vq_loss:.{6}f}",
                         "num": f"{self.gaussian_model.cur_num_points}"})
                    progress_bar.update(100)
                if self.args.prune:
                    none_definite = 0
                    if iter % self.args.prune_iter == 0 and iter < self.args.warmup_iter:
                        none_definite, cur_gs_nums = self.gaussian_model.non_semi_definite_prune(self.H, self.W)
                        if none_definite:
                            self.logwriter.write(
                                f"iter {iter}, pruned {none_definite} quantized-points due to non positive definite, cur_gs_nums:{cur_gs_nums},psnr:{psnr}")
                if best_psnr < psnr:
                    best_psnr = psnr
                    best_model_dict = copy.deepcopy(self.gaussian_model.state_dict())
                    slv_bound = copy.deepcopy(self.gaussian_model.cholesky_bound)
                    best_iter = iter

                if self.args.adaptive_add:
                    if iter % self.args.grow_iter == 0 and iter < self.args.warmup_iter:
                        new_added_points_nums = self.add_sample_positions(out_image, iter=iter)

        none_definite, cur_gs_nums = self.gaussian_model.non_semi_definite_prune(self.H, self.W)
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
        # self.gaussian_model.sort_morton()

        best_psnr_value, best_ms_ssim_value, best_vq_bpp, test_end_time, FPS = self.test(best=True)
        print("save to ", self.log_dir)
        des_path = os.path.join(self.log_dir, self.image_name)
        os.makedirs(des_path, exist_ok=True)
        torch.save({"gs": best_model_dict,
                    "num_gs": self.gaussian_model.cur_num_points,
                    "psnr": best_psnr, "ms-ssim": best_ms_ssim_value, "slv_bound": slv_bound
                    }, os.path.join(des_path, "gaussian_model.best.pth.tar"))

        self.logwriter.write(
            "{} Training Complete in {:.4f}s, Eval time:{:.8f}s, FPS:{:.4f}, best_iter: {} best_psnr:{:.4f},best_ms-ssim:{:.4f}".format(
                self.image_name, end_time, test_end_time, FPS,
                best_iter, best_psnr_value, best_ms_ssim_value))

        data_dict = self.encode()
        data_dict['training_time'] = end_time
        return data_dict

    def test(self, best=False):
        self.gaussian_model.eval()
        with torch.no_grad():
            out = self.gaussian_model.forward_quantize()
            out_img = out["render"].float()
            mse_loss = F.mse_loss(out_img, self.gt_image.float())
            psnr = 10 * math.log10(1.0 / mse_loss.item())
            ms_ssim_value = ms_ssim(out["render"].float(), self.gt_image.float(), data_range=1,
                                    size_average=True).item()
            m_bits, s_bit, c_bit = out["unit_bit"]
            bpp = (m_bits + s_bit + c_bit) / self.H / self.W
            bpp = bpp.item() if isinstance(bpp, torch.Tensor) else bpp

            torch.cuda.synchronize()
            test_start_time = time.time()
            for i in range(100):
                _ = self.gaussian_model.forward_quantize()
            torch.cuda.synchronize()
            test_end_time = (time.time() - test_start_time) / 100

        strings = "Best Test" if best else "Test"
        self.logwriter.write("{} PSNR:{:.4f}, MS_SSIM:{:.6f}, vq_bpp:{:.4f} FPS:{:.4f}".format(strings, psnr,
                                                                                               ms_ssim_value, bpp,
                                                                                               1 / test_end_time))

        if self.save_imgs:
            transform = transforms.ToPILImage()
            img = transform(out_img.squeeze(0))
            name = "_codec_best.png" if best else "_codec.png"
            name = self.image_name + name
            img.save(str(self.log_dir / name))
        return psnr, ms_ssim_value, bpp, test_end_time, 1 / test_end_time

    def encode(self):
        self.gaussian_model.eval()
        with torch.no_grad():
            encoding_dict = self.gaussian_model.compress_wo_ec()
            out = self.gaussian_model.decompress_wo_ec(encoding_dict)
            start_time = time.time()
            for i in range(100):
                _ = self.gaussian_model.decompress_wo_ec(encoding_dict)
            end_time = (time.time() - start_time) / 100

        data_dict = self.gaussian_model.analysis_wo_ec(encoding_dict)
        data_dict['cholesky_bpp_wc'] = compress_matrix_flatten_gaussian_global(encoding_dict["quant_cholesky_elements"].float()) / self.H / self.W
        data_dict['feature_dc_bpp_wc'] = compress_matrix_flatten_gaussian_global( encoding_dict["feature_dc_index"].float()) / self.H / self.W
        data_dict['bpp_wc'] = data_dict['position_bpp'] + data_dict['cholesky_bpp_wc'] + data_dict['feature_dc_bpp_wc']
        out_img = out["render"].float()
        mse_loss = F.mse_loss(out_img, self.gt_image)
        psnr = 10 * math.log10(1.0 / mse_loss.item())
        ms_ssim_value = ms_ssim(out_img, self.gt_image, data_range=1, size_average=True).item()

        data_dict["psnr"] = psnr
        data_dict["ms-ssim"] = ms_ssim_value
        data_dict["rendering_time"] = end_time
        data_dict["rendering_fps"] = 1 / end_time
        data_dict['gs_nums'] = self.gaussian_model.cur_num_points

        self.logwriter.write(
            "{} Eval time:{:.8f}s, FPS:{:.4f} PSNR:{:.4f}, MS_SSIM:{:.6f}, bpp:{:.4f} position_bpp:{:.4f}, cholesky_bpp:{:.4f}, feature_dc_bpp:{:.4f}".format(
                self.image_name, end_time, 1 / end_time, psnr, ms_ssim_value, data_dict["bpp"],
                data_dict["position_bpp"], data_dict["cholesky_bpp"], data_dict["feature_dc_bpp"]))

        return data_dict




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
        "--prune_iter", type=int, default=100, help="number of training epochs (default: %(default)s)"
    )
    parser.add_argument(
        "--grow_iter", type=int, default=5000, help="number of training epochs (default: %(default)s)"
    )
    parser.add_argument(
        "--model_name", type=str, default="GaussianImage_Covariance",
        help="model selection: GaussianImage_Cholesky, GaussianImage_RS, 3DGS"
    )
    parser.add_argument(
        "--sh_degree", type=int, default=3, help="SH degree (default: %(default)s)"
    )
    parser.add_argument("--num_points", type=int, default=2500, help="2D GS points (default: %(default)s)")
    parser.add_argument("--max_num_points", type=int, default=5000, help="2D GS points (default: %(default)s)", )
    parser.add_argument("--opt_type", type=str, default="adam", help="the nums of optimizer")
    parser.add_argument("-opt", "--opt_nums", type=int, default=1, help="the nums of optimizer")
    parser.add_argument("--model_path", type=str, default=None, help="Path to a checkpoint")
    parser.add_argument("--seed", type=int, default=3047, help="Set random seed for reproducibility")
    parser.add_argument("--save_imgs", action="store_true", help="Save image")
    parser.add_argument(
        "--lr",
        type=float,
        default=0.018,
        # default=0.0001, #better
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument("--warmup_iter", type=float, default=6000, help="tanh")
    parser.add_argument('--radius_clip', type=float, default=1.0)
    parser.add_argument("--prune", type=bool, default=True, help="Set random seed for reproducibility")
    parser.add_argument("--adaptive_add", type=bool, default=True, help="Set random seed for reproducibility")
    parser.add_argument("--wandb-project", type=str, default=None, help='Weights & Biases Project')
    parser.add_argument("--loss_type", type=str, default="L2", help="Set random seed for reproducibility")
    parser.add_argument("--SLV_init", type=bool, default=True, help="if the background is learned or not")
    parser.add_argument("--color_norm", type=bool, default=True)
    parser.add_argument("--coords_norm", action='store_true', help="if normalize the coordinates")
    parser.add_argument("--coords_act", type=str, default="tanh", help="activate function of coordinates normalization")
    parser.add_argument("--save_interval", type=int, default=5, help="tanh")
    parser.add_argument("--clip_coe", type=float, default=3., help="Set random seed for reproducibility")
    parser.add_argument("--pretrained", type=str, help="Path to a checkpoint")

    # =============== quantizzaton params ====================
    parser.add_argument("--quantize", type=bool, default=False, help="Quantize")
    parser.add_argument("--cov_quant", type=str, default="lsq", help="type of covariance quantization")
    parser.add_argument("--color_quant", type=str, default="lsq")
    parser.add_argument("--xy_quant", type=str, default="lsq")
    parser.add_argument("--xy_bit", type=int, default=12, help="bitdepth of xy attri")
    parser.add_argument("--cov_bit", type=int, default=10, help="bitdepth of cov attri")
    parser.add_argument("--color_bit", type=int, default=6, help="bitdepth of color attri")

    args = parser.parse_args(argv)
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    log_dir = (
            f"checkpoints_quant/{args.data_name}/{args.model_name}_I{args.iterations}_N{args.num_points}_joint_{'_SLV' if args.SLV_init else ''}{'_coordsnorm' if args.coords_norm else ''}" +
            f"{args.loss_type if args.loss_type != 'L2' else ''}{'_prune' if args.prune else ''}{'_color_norm' if args.color_norm else ''}"
    )
    logwriter = LogWriter(Path(log_dir))
    script_name = os.path.basename(__file__)
    logwriter.write(script_name)
    logwriter.write(args_text)
    psnrs, ms_ssims, training_times, eval_times, eval_fpses, bpps, bpps_wc, gs_nums, params = [], [], [], [], [], [], [], [], []
    best_psnrs, best_ms_ssims, best_bpps = [], [], []
    position_bpps, cholesky_bpps, feature_dc_bpps = [], [], []
    image_h, image_w = 0, 0
    image_length, start = 1, 0
    if args.data_name == "DIV2K_valid_HR":
        image_length, start = 100, 800

    model_path = (
            f"./checkpoints/{args.data_name}/{args.model_name}_I{args.iterations}_N{args.num_points}{'_SLV' if args.SLV_init else ''}_R{args.radius_clip}{'_add' if args.adaptive_add else ''}" +
            f"{'_prune' if args.prune else ''}{'_colornorm' if args.color_norm else ''}"
    )
    for i in range(start, start + image_length):
        image_path = Path(args.dataset) / f'kodim{i + 1:02}.png'
        model_path = Path(model_path) / f'kodim{i + 1:02}' / 'gaussian_model.pth.tar' #if args.model_path is not None else None

        if args.data_name == "DIV2K_valid_HR":
            image_path = Path(args.dataset) / f'{i + 1:04}.png'
            model_path = Path(model_path) / f'{i + 1:04}' / 'gaussian_model.pth.tar'


        trainer = SimpleTrainer2d(image_path=image_path, num_points=args.num_points,
                                  iterations=args.iterations, args=args,
                                  model_path=model_path,
                                  log_dir=log_dir)

        data_dict = trainer.train()
        psnrs.append(data_dict["psnr"])
        ms_ssims.append(data_dict["ms-ssim"])
        eval_times.append(data_dict["rendering_time"])
        eval_fpses.append(data_dict["rendering_fps"])
        bpps.append(data_dict["bpp"])
        bpps_wc.append(data_dict["bpp_wc"])
        position_bpps.append(data_dict["position_bpp"])
        cholesky_bpps.append(data_dict["cholesky_bpp"])
        feature_dc_bpps.append(data_dict["feature_dc_bpp"])
        finally_gs_nums = data_dict['gs_nums']
        finally_params = sum([p.numel() for p in trainer.gaussian_model.parameters() if p.requires_grad])
        gs_nums.append(finally_gs_nums)
        params.append(finally_params / 1e6)
        training_times.append(data_dict['training_time'])
        image_h += trainer.H
        image_w += trainer.W
        image_name = image_path.stem
        logwriter.write(
            "{}: {}x{}, PSNR:{:.4f}, MS-SSIM:{:.4f}, bpp:{:.4f}, bpp_wc:{:.4f}, Eval:{:.6f}s, FPS:{:.4f}, Train:{:.6f}s, position_bpp:{:.4f}, cholesky_bpp:{:.4f}, feature_dc_bpp:{:.4f} gs_nums:{}".format(
                image_name, trainer.H, trainer.W, data_dict["psnr"], data_dict["ms-ssim"], data_dict["bpp"],
                data_dict["bpp_wc"],
                data_dict["rendering_time"], data_dict["rendering_fps"], data_dict['training_time'],
                data_dict["position_bpp"], data_dict["cholesky_bpp"], data_dict["feature_dc_bpp"],
                trainer.gaussian_model.cur_num_points))

    avg_psnr = torch.tensor(psnrs).mean().item()
    avg_ms_ssim = torch.tensor(ms_ssims).mean().item()
    avg_training_time = torch.tensor(training_times).mean().item()
    avg_eval_time = torch.tensor(eval_times).mean().item()
    avg_eval_fps = torch.tensor(eval_fpses).mean().item()
    avg_bpp = torch.tensor(bpps).mean().item()
    avg_bpp_wc = torch.tensor(bpps_wc).mean().item()
    avg_position_bpp = torch.tensor(position_bpps).mean().item()
    avg_cholesky_bpp = torch.tensor(cholesky_bpps).mean().item()
    avg_feature_dc_bpp = torch.tensor(feature_dc_bpps).mean().item()
    avg_best_psnr = torch.tensor(best_psnrs).mean().item()
    avg_best_ms_ssim = torch.tensor(best_ms_ssims).mean().item()
    avg_best_bpp = torch.tensor(best_bpps).mean().item()
    avg_h = image_h // image_length
    avg_w = image_w // image_length
    avg_gs_nums = sum(gs_nums) / image_length
    avg_params = sum(params) / image_length
    logwriter.write(
        "Average: {}x{}, PSNR:{:.4f}, MS-SSIM:{:.4f}, bpp:{:.4f},bpp_wc:{:.4f},train_times:{:.8f}s, Eval:{:.8f}s, FPS:{:.4f}, position_bpp:{:.4f}, cholesky_bpp:{:.4f}, feature_dc_bpp:{:.4f}  gs_nums:{:.2e}, Params(M):{:.2f}".format(
            avg_h, avg_w, avg_psnr, avg_ms_ssim, avg_bpp, avg_bpp_wc, avg_training_time, avg_eval_time, avg_eval_fps,
            avg_position_bpp, avg_cholesky_bpp, avg_feature_dc_bpp, avg_gs_nums, avg_params))
    # logwriter.write("Average: {}x{}, PSNR:{:.4f}, MS-SSIM:{:.4f}, Bpp:{:.4f}, Best PSNR:{:.4f}, Best MS-SSIM:{:.4f}, Best bpp:{:.4f}, Training:{:.4f}s, Eval:{:.8f}s, FPS:{:.4f}, gs_nums:{:.2e}, Params(M):{:.2f}".format(
    #     avg_h, avg_w, avg_psnr, avg_ms_ssim, avg_bpp, avg_best_psnr, avg_best_ms_ssim, avg_best_bpp, avg_training_time, avg_eval_time, avg_eval_fps,  avg_gs_nums,avg_params))      


if __name__ == "__main__":
    main(sys.argv[1:])
