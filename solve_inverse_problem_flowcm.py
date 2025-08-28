import copy
import os
import yaml
import random 
import tqdm
import argparse
import collections
import torch
import torch.optim as optim
import torch.nn.functional as F
from absl import app, flags
import torchvision
import torchvision.utils as tvu
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ExponentialLR
# import dist_util
from skimage.metrics import structural_similarity
# import lpips
import time
from sam import SAM
from sampler import karras_sample, KarrasDenoiser, get_generator
from ms_ssim_loss import MS_SSIM_L1_LOSS
from models.unet.unet import UNetModelWrapper
from models.unet.unet_lsun import UNetModel
from models.unet.unet_act import UNet2DModel
from img_datasets.image_datasets import get_val_dataset, data_transform, inverse_data_transform
from utils.utils import plot_our_sampler_metrics
import numpy as np
from degradations.operators import color2gray, gray2color, MeanUpsample, MotionBlurOperator, GaussialBlurOperator, NonlinearBlurOperator
from degradations.phase_retrieval import fft2_m, ifft2_m
import torch.utils.data as data


CALCULATE_TIME=True
FLAGS = flags.FLAGS
flags.DEFINE_integer("num_workers", 8, help="workers of Dataloader")

# Model related args
flags.DEFINE_string("model", "fm", help="cm")
flags.DEFINE_string("ckpt_path", "./", help="ckpt_dir")
flags.DEFINE_string("exp_dir", "./inverse_problem_results", help="results_dir for inverse problem results")

# Dataset & Dataloader
flags.DEFINE_string("dataset", "afhq", help="Datset for inverse problem solving")
flags.DEFINE_string("data_dir", "./data/afhq_v2/test", help="Test datset path for inverse problem solving")
flags.DEFINE_string("data_config_pth", "./configs/afhq.yml", help="yml file for data config")
flags.DEFINE_integer("batch_size", 1, help="batch size")

# UNet
flags.DEFINE_integer("num_channel", 128, help="base channel of UNet")

# Inverse problem related args
flags.DEFINE_integer("seed", 44, help="seed")
flags.DEFINE_string("deg", "inpaint", help="Degradation selector")
flags.DEFINE_float("deg_scale", 0.1, help="degradation scale")
flags.DEFINE_float("sigma", 0.05, help="noise level") 
flags.DEFINE_string("mask_path", "./", help="path for obtaining pre-determined masks")

# Sampling related args
flags.DEFINE_string("sampling_method", "fps", help="Supported methods include fps")
flags.DEFINE_integer("num_plot", 5, help="number of recons to plot")
flags.DEFINE_integer("num_per_image", 1, help="number of recons to get for each image")

flags.DEFINE_string("init_type", "Apy", help="Supported methods include rand | Apy | gt | warm | blend | filepath")
flags.DEFINE_string("warm_init_type", "rand", help="Supported methods include rand | Apy | gt | blend")
flags.DEFINE_string("warm_solver", "GD", help="Supported warm start solvers include GD | LBFGS")
flags.DEFINE_float("warm_lr", 0.001, help="Learning rate of warm start solver")
flags.DEFINE_float("warm_momentum", 0.9, help="Momentum of warm start solver")
flags.DEFINE_integer("num_warm_steps", 10, help="Number of warm start solver steps")
flags.DEFINE_float("blend_alpha", 0.01, help="Blending coefficient for init")

flags.DEFINE_string("cm_solver", "multistep", help="Solver for consistency model can be multistep | onestep")
flags.DEFINE_integer("numsteps_cm", 5, help="Number of sampling steps for consistency model")
flags.DEFINE_integer("num_diffusion_steps", 40, help="Number of diffusion steps")

# LBFGS solver related args
flags.DEFINE_integer("warm_hist", 10, help="LBFGS history size")
flags.DEFINE_integer("warm_max_iter", 20, help="Max num internal LBFGS solver steps")
flags.DEFINE_float("max_g_norm", 10000., help="Max norm of gradient for norm clipping")

# args related to fps
flags.DEFINE_string("sde_solver", "EM", help="Supported SDE solvers include EM | EI | GD")
flags.DEFINE_float("tau", 0.1, help="Step size of Langevin dynamics")
flags.DEFINE_float("cond_sigma", 0.05, help="Variance of degradation operator")
flags.DEFINE_integer("num_lang_steps", 100, help="Number of steps to conduct Langevin sampling")

# For fidelity and diversity trade-off during sampling using dynamic tau
flags.DEFINE_bool("dynamic_tau", False, help="whether to use dynamic Tau or not. This tries to prioritize both fidelity and diversity by switching tau")
flags.DEFINE_integer("dynamic_tau_history", 5, help="Max num samples before switcing on dynamic tau")

# args related to regularization (for warm start of fps)
flags.DEFINE_string("regularizer", "none", help="Supported regularizers include none | chi")
flags.DEFINE_float("reg_lam", 0.01, help="Weight of regularization")

flags.DEFINE_bool("verbose", True, help="Show tqdm progress bar") 
flags.DEFINE_bool("decay_tau", False, help="Use variable step size for EM sampler") 
flags.DEFINE_bool("reinit_gen", True, help="Reinitialize generator to fix the noise addition during multi-step sampling") 

flags.DEFINE_integer("plot_start_index", 0, help="Index of validation set")
flags.DEFINE_bool("plot_all", False, help="plot all intermediate samples of diffusion") 

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

flags.DEFINE_string("user_exp_suffix", "", help="User suffix while creating dir")


def average_image_patches(x, image_size, patch_size):
    x_flatten = (
        x.reshape(-1, 3, image_size, image_size)
        .reshape(
            -1,
            3,
            image_size // patch_size,
            patch_size,
            image_size // patch_size,
            patch_size,
        )
        .permute(0, 1, 2, 4, 3, 5)
        .reshape(-1, 3, image_size**2 // patch_size**2, patch_size**2)
    )
    x_flatten[..., :] = x_flatten.mean(dim=-1, keepdim=True)
    return (
        x_flatten.reshape(
            -1,
            3,
            image_size // patch_size,
            image_size // patch_size,
            patch_size,
            patch_size,
        )
        .permute(0, 1, 2, 4, 3, 5)
        .reshape(-1, 3, image_size, image_size)
    )

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def setup_seed_and_cuda(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def export_flags_to_yaml(file_path):
    # Convert FLAGS to a dictionary
    flags_dict = {name: FLAGS[name].value for name in FLAGS}
    with open(file_path, 'w') as file:
        yaml.dump(flags_dict, file, default_flow_style=False)

def setup_experiment_directory():
    # Construct the directory name
    main_exp_dir = FLAGS.exp_dir
    dir_name = "{}_{}_{}_sigma{:.1e}_{}".format(FLAGS.dataset, FLAGS.deg, FLAGS.deg_scale, FLAGS.sigma, FLAGS.model)
    base_dir = os.path.join(main_exp_dir, dir_name)

    # Make directories if they do not exist
    os.makedirs(base_dir, exist_ok=True)

    sub_dirs = ["Apy", "gt", "y"]
    for sub_dir in sub_dirs:
        path_ = os.path.join(base_dir, sub_dir)
        os.makedirs(path_, exist_ok=True)
   
    return base_dir


class SolveInverseProblem(object):
    def __init__(self, base_dir, config, device=None):
        """
        include model init and ckpt loading

        """
        self.base_exp_dir = base_dir
        self.config = config
        self.dataset = FLAGS.dataset
        self.dataset_pth = FLAGS.data_dir
        self.degradation = FLAGS.deg
        self.deg_scale = FLAGS.deg_scale
        self.ckpt_path = FLAGS.ckpt_path
        self.device = device
        self.diffusion = None
        # Dataloader Setup
        _, test_dataset = get_val_dataset(self.dataset_pth, config)

        if FLAGS.dataset == "lsun_bedroom" or FLAGS.dataset == "lsun_cat":
            self.img_size = 256
        elif FLAGS.dataset == "imagenet64":
            self.img_size = 64
        elif FLAGS.dataset == "cifar10":
            self.img_size = 32
    
        def seed_worker(worker_id):
            worker_seed = FLAGS.seed % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(FLAGS.seed)
        self.val_loader = data.DataLoader(
            test_dataset,
            batch_size=FLAGS.batch_size,
            shuffle=False,
            num_workers=FLAGS.num_workers,
            # worker_init_fn=seed_worker,
            generator=g,
        )

        # Degradation Setup
        if "inpaint" in self.degradation:
            if self.degradation == "square_mask_inpaint":
                loaded = np.load("./inp_masks/mask_square.npy")
                mask = torch.from_numpy(loaded).to(self.device)
                self.A = lambda z: z*mask
                self.Ap = self.A
            elif self.degradation == "random_pixel_inpaint":
                # Currently only [1%, 2.5%, 5.0%, 10.0% and 20.0%] pixel visibility
                loaded = np.load(os.path.join(FLAGS.mask_path, "random_pixel", f"{self.deg_scale}_percent", "mask.npy")).astype(np.float32)
                mask = torch.from_numpy(loaded).to(self.device)
                self.A = lambda z: z*mask
                self.Ap = self.A
            else:
                raise NotImplementedError
            
        elif self.degradation == "colorization":
            self.A = lambda z: color2gray(z)
            self.Ap = lambda z: gray2color(z)

        elif self.degradation == "gaussian_denoising":
            self.A = lambda z: torch.clone(z)
            self.Ap = self.A
        
        elif self.degradation == "poisson_denoising":
            self.A = lambda z: torch.clone(z)
            self.Ap = self.A

        elif self.degradation == "sr_avgpooling":
            scale=round(self.deg_scale)
            self.A = torch.nn.AdaptiveAvgPool2d((self.img_size//scale,self.img_size//scale))
            self.Ap = lambda z: F.interpolate(z, scale_factor=scale, mode='bicubic')

        elif self.degradation == "avg_patch_downsampling":
            scale=round(self.deg_scale)
            self.A = average_image_patches
            self.Ap = lambda z: z

        elif self.degradation == "motion_blur":
            raise NotImplementedError

        elif self.degradation == "gaussian_deblur":
            self.gaussian_blr_op = GaussialBlurOperator(kernel_size=61, intensity=3.0, device=device)
            self.A = lambda z : self.gaussian_blr_op.forward(z)
            self.Ap = lambda z: torch.clone(z)
            
        elif self.degradation == "non_linear_blur":
            self.nl_blur_op = NonlinearBlurOperator("bkse/options/generate_blur/default.yml", device)
            self.A = lambda z : self.nl_blur_op.forward(z)
            self.Ap = lambda z: torch.clone(z)

        elif self.degradation == "hdr":
            self.A = lambda z : torch.clip((z * self.deg_scale), -1, 1)
            self.Ap = lambda z: torch.clone(z) / self.deg_scale

        elif self.degradation == "non_uniform_gaussian_noise":
            raise NotImplementedError

        elif self.degradation == "phase_retrieval":
            pad = int((self.deg_scale / 8.0) * 256)
            self.A = lambda z: fft2_m(F.pad(z, (pad, pad, pad, pad))).abs()
            self.Ap = lambda z: ifft2_m(z).abs()

        else:
            raise NotImplementedError("Degradation type not found....")
        
        checkpoint = torch.load(self.ckpt_path, map_location=self.device)
        
        # Model setup and ckpt loading 
        if FLAGS.dataset == "lsun_bedroom" or FLAGS.dataset == "lsun_cat":
            self.rescale_t = True
            self.net_model = UNetModel(
                image_size=256,
                in_channels=3,
                model_channels=256,
                out_channels=3,
                num_res_blocks=2,
                attention_resolutions=(32,16,8),
                dropout=0.0,
                channel_mult=(1, 1, 2, 2, 4, 4),
                num_classes=None,
                use_checkpoint=False,
                use_fp16=True,
                num_heads=4,
                num_head_channels=64,
                num_heads_upsample=-1,
                use_scale_shift_norm=False,
                resblock_updown=True,
                use_new_attention_order=False,
            )

            self.net_model.load_state_dict(checkpoint, strict=True)

        elif FLAGS.dataset == "imagenet64":
            self.rescale_t = True
            self.net_model = UNetModel(
                image_size=64,
                in_channels=3,
                model_channels=192,
                out_channels=3,
                num_res_blocks=3,
                attention_resolutions=(32,16,8,4,2),
                dropout=0.0,
                channel_mult=(1, 2, 3, 4),
                num_classes=1000,
                use_checkpoint=False,
                use_fp16=True,
                num_heads=4,
                num_head_channels=64,
                num_heads_upsample=-1,
                use_scale_shift_norm=True,
                resblock_updown=True,
                use_new_attention_order=False,
            )

            self.net_model.load_state_dict(checkpoint, strict=True)

        elif FLAGS.dataset == "cifar10":
            self.rescale_t = False
            self.net_model = UNet2DModel(
                sample_size=32,
                in_channels=3,
                out_channels=6,
                layers_per_block=2,
                block_out_channels=(128, 128, 256, 256),
                down_block_types=(
                    "DownBlock2D",
                    "AttnDownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D"
                ),
                up_block_types=(
                    "UpBlock2D",
                    "UpBlock2D",
                    "AttnUpBlock2D",
                    "UpBlock2D",
                ),
            )

            new_state_dict = collections.OrderedDict()
            for key in checkpoint['state_dict'].keys():
                if key[:11] == "model.unet.":
                    new_state_dict[key[11:]] = checkpoint['state_dict'][key]
            self.net_model.load_state_dict(new_state_dict, strict=True)

        else:
            raise NotImplementedError("Dataset not supported...")

        self.sigma_data = 0.5
        self.sigma_max = 80.0
        self.sigma_min = 0.002

        self.steps = FLAGS.num_diffusion_steps

        self.generator = get_generator("determ", 1, 42)
        
        self.diffusion = KarrasDenoiser(
                sigma_data=self.sigma_data,
                sigma_max=self.sigma_max,
                sigma_min=self.sigma_min,
                distillation=True,
                weight_schedule="uniform",
        )
        
        self.net_model.to(self.device)
        
        if FLAGS.dataset == "lsun_bedroom" or FLAGS.dataset == "lsun_cat" or FLAGS.dataset == "imagenet64":
            self.net_model.convert_to_fp16()
            self.net_model.to(self.device)
            
        # ms-ssim loss
        self.loss_func = MS_SSIM_L1_LOSS()


    def sample_true(self, x0, plot_all=False, plot_name_base=None, label=None, stage=None, step_num=None):
        # sampling using consistency model
        generator = get_generator("determ", 1, 42)
        
        if FLAGS.dataset == "imagenet64":
            label = label.to(self.device)
            model_kwargs = {'y':label}
        else:
            model_kwargs = {}
        ts = tuple(np.linspace(0,self.steps-1,FLAGS.numsteps_cm,dtype=int))
        
        sample = karras_sample(
            self.diffusion,
            self.net_model,
            (1, 3, self.img_size, self.img_size),
            steps=self.steps,
            model_kwargs=model_kwargs,
            device=self.device,
            clip_denoised=True,
            sampler=FLAGS.cm_solver,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            s_churn=0.0,
            s_tmin=0.0,
            s_tmax="inf",
            s_noise=1.0,
            # generator=self.generator,
            generator=generator,
            ts=ts,
            x_initial = x0,
            plot_all=plot_all,
            plot_name_base=plot_name_base,
            t_rescale=self.rescale_t,
            c_stage = stage,
            step_num = step_num
        )
       
        return sample

    def conditional_sample(self):
        """
        Dataset, degradation and sampling
        """
        self.flow_posterior_sampler()
        
    
    def flow_posterior_sampler(self):
        idx_so_far = 0

        if FLAGS.batch_size!=1:
            raise ValueError("please set batch size to 1 for inverse problems")


        recon_dir_name = "fps_{}_".format(FLAGS.init_type)
        if FLAGS.init_type == "warm":
            recon_dir_name += "{}_{:.1e}_{}_".format(FLAGS.warm_solver, FLAGS.warm_lr, FLAGS.num_warm_steps)
            if FLAGS.warm_solver == "LBFGS":
                recon_dir_name += "warmmaxiter{:d}_warmhist{:d}_".format(FLAGS.warm_max_iter, FLAGS.warm_hist)
            elif FLAGS.warm_solver == "SAM":
                recon_dir_name += "momentum{:.1f}_".format(FLAGS.warm_momentum)
            if FLAGS.regularizer != "none":
                recon_dir_name += "{}_lam{:.1e}_".format(FLAGS.regularizer, FLAGS.reg_lam)
        assert FLAGS.sde_solver == "EM"
        recon_dir_name += "{}_cond{:.1e}_tau{:.1e}_langstep{}".format(
            FLAGS.sde_solver, FLAGS.cond_sigma, FLAGS.tau, FLAGS.num_lang_steps
            )
        recon_dir_name += "_CM_"
        recon_dir_name += "steps{}".format(self.steps)
        if FLAGS.cm_solver != "onestep":
            recon_dir_name += "_cmsteps{}".format(FLAGS.numsteps_cm)
        recon_dir_name += "_{}".format(FLAGS.cm_solver)
        
        if FLAGS.user_exp_suffix:
            recon_dir_name += "_{}".format(FLAGS.user_exp_suffix)
        
        if FLAGS.decay_tau:
            recon_dir_name += "_decay_tau"
        print(recon_dir_name)
        os.makedirs(os.path.join(self.base_exp_dir, recon_dir_name), exist_ok=True)
        export_flags_to_yaml(os.path.join(os.path.join(self.base_exp_dir, recon_dir_name), "flags.yaml"))

        # make wram_up dir and samples dir 
        os.makedirs(os.path.join(self.base_exp_dir, recon_dir_name, 'warm_up'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, recon_dir_name, 'samples'), exist_ok=True)
        pbar = tqdm.tqdm(self.val_loader) if FLAGS.verbose else self.val_loader
        
        if CALCULATE_TIME:
            start_time = time.time()
            
        for x_orig, label in pbar:
            if idx_so_far < FLAGS.plot_start_index:
                idx_so_far += 1
                if idx_so_far >= FLAGS.num_plot + FLAGS.plot_start_index:
                    break
                else:
                   
                    continue
            
            # print("Running for :", idx_so_far)
            x_orig = x_orig.to(self.device)
            x_orig = data_transform(x_orig) 

            if self.degradation == "avg_patch_downsampling":
                y = self.A(x_orig, 256, round(FLAGS.deg_scale))
            else:
                y = self.A(x_orig).detach()
            y += torch.randn_like(y)*FLAGS.sigma

            if not FLAGS.plot_all:
                if os.path.exists(os.path.join(self.base_exp_dir, f"{recon_dir_name}/fps_{idx_so_far}_repeat{FLAGS.num_per_image-1}.png")):
                    idx_so_far += y.shape[0]
                    continue

            Apy = self.Ap(y)

            for i in range(len(Apy)):
                tvu.save_image(
                    inverse_data_transform(y[i]),
                    os.path.join(self.base_exp_dir, f"y/y_{idx_so_far + i}.png")
                )
                tvu.save_image(
                    inverse_data_transform(Apy[i]),
                    os.path.join(self.base_exp_dir, f"Apy/Apy_{idx_so_far + i}.png")
                )
                tvu.save_image(
                    inverse_data_transform(x_orig[i]),
                    os.path.join(self.base_exp_dir, f"gt/orig_{idx_so_far + i}.png")
                )


            if FLAGS.init_type == "rand":
                shape = x_orig.shape
                x0 = self.generator.randn(*shape, device=self.device).detach()

            elif FLAGS.init_type == "warm":
                shape = x_orig.shape
                x0 = self.generator.randn(*shape, device=self.device).detach()

                warm_loss = []

                if FLAGS.warm_solver == "GD" or FLAGS.warm_solver == "ADAM":
                    # Initialize Adam optimizer
                    if FLAGS.warm_solver == "ADAM":
                        optimizer = optim.Adam([x0], lr=FLAGS.warm_lr)
                        # optimizer = optim.Adagrad([x0], lr=FLAGS.warm_lr)
                        # scheduler = ExponentialLR(optimizer, gamma=0.9)
                        
                    for _warm_step in range(FLAGS.num_warm_steps + 1):
                        x0 = x0.requires_grad_()
                        x1_hat = self.sample_true(x0, label=label, stage="optim" , step_num=_warm_step)

                        if _warm_step != FLAGS.num_warm_steps - 1:
                            if FLAGS.warm_solver == "GD":
                                L = torch.pow(torch.linalg.norm(y - self.A(x1_hat)), 2)
                                
                                if FLAGS.regularizer == "none":
                                    L_grad = torch.autograd.grad(outputs=L.sum(), inputs=x0)[0]
                                elif FLAGS.regularizer == "chi":
                                    _L = L + FLAGS.reg_lam * ((x0.numel() - 1) * torch.log10(torch.linalg.norm(x0)) - 0.5 * torch.pow(torch.linalg.norm(x0), 2))
                                    L_grad = torch.autograd.grad(outputs=_L.sum(), inputs=x0)[0]

                                x0 = x0.detach_()
                                L = L.detach()
                                L_grad = L_grad.detach()

                                g = L_grad
                                gd_term = -1 * FLAGS.warm_lr * g
                                x0 = x0 + gd_term
                            elif FLAGS.warm_solver == "ADAM":
                                # Zero gradients for Adam
                                optimizer.zero_grad()
                                
                                # Recompute the loss for the current step as Adam expects a backward pass
                                if FLAGS.regularizer == "none":
                                    L = torch.pow(torch.linalg.norm(y - self.A(x1_hat)), 2)
                                elif FLAGS.regularizer == "chi":
                                    L = L + FLAGS.reg_lam * ((x0.numel() - 1) * torch.log10(torch.linalg.norm(x0)) - 0.5 * torch.pow(torch.linalg.norm(x0), 2))
                                # print("using MS-SSIM losss..")
                                # L = self.loss_func(y, self.A(x1_hat))
                                
                                L.backward()  # Compute gradients
                                optimizer.step()  # Adam updates the parameters

                            warm_loss.append(L.detach().cpu())
                            # if _warm_step % 100 == 0 and FLAGS.warm_solver == "ADAM":
                            #     scheduler.step()
                                
                        if FLAGS.plot_all:
                            for i in range(len(x1_hat)):
                                tvu.save_image(
                                    inverse_data_transform(x1_hat[i]), os.path.join(self.base_exp_dir, f"{recon_dir_name}/warm_up/warm_{idx_so_far + i}_step{_warm_step}.png")
                                )

                elif FLAGS.warm_solver == "SAM":
                    x0 = x0.requires_grad_()

                    base_optimizer = torch.optim.Adam  # define an optimizer for the "sharpness-aware" update
                    optimizer = SAM([x0], base_optimizer, lr=FLAGS.warm_lr)

                    warm_loss = []

                    def closure():
                        x1_hat = self.sample_true(x0, label=label, stage="optim")

                        if FLAGS.plot_all:
                            tvu.save_image(
                              inverse_data_transform(x1_hat), os.path.join(self.base_exp_dir, f"{recon_dir_name}/warm_{idx_so_far + i}_step{_warm_step}.png")
                              )
                        
                        L = torch.pow(torch.linalg.norm(y - self.A(x1_hat)), 2)
                        warm_loss.append(L.detach().cpu())
                        L.backward()
                        return L
                    
                    for _warm_step in range(1,FLAGS.num_warm_steps+1):
                        x1_hat = self.sample_true(x0, label=label, stage="optim")
                        L = torch.pow(torch.linalg.norm(y - self.A(x1_hat)), 2)
                        L.backward()
                        optimizer.step(closure)
                        optimizer.zero_grad()

                elif FLAGS.warm_solver == "LBFGS":
                    x0 = x0.requires_grad_()

                    lbfgs = optim.LBFGS([x0],
                            lr=FLAGS.warm_lr,
                            history_size=FLAGS.warm_hist, 
                            max_iter=FLAGS.warm_max_iter,
                            line_search_fn="strong_wolfe")

                    def closure():
                        lbfgs.zero_grad()
                        
                        x1_hat = self.sample_true(x0, label=label)
                        
                        if FLAGS.plot_all:
                            tvu.save_image(
                              inverse_data_transform(x1_hat), os.path.join(self.base_exp_dir, f"{recon_dir_name}/warm_{idx_so_far + i}_step{_warm_step}.png")
                              )

                        L = torch.pow(torch.linalg.norm(y - self.A(x1_hat)), 2)
                        warm_loss.append(L.detach().cpu())
                        if FLAGS.regularizer == "chi":
                            L += FLAGS.reg_lam * ((x0.numel()-1)*torch.log10(torch.linalg.norm(x0)) - 0.5*torch.pow(torch.linalg.norm(x0),2))
                        L.backward()
                        return L
                    
                    for _warm_step in range(1,FLAGS.num_warm_steps+1):
                        lbfgs.step(closure)

                if FLAGS.plot_all:
                    warm_loss = torch.stack(warm_loss, dim=0)
                    plt.figure(figsize=(10, 6))  
                    plt.plot(warm_loss, c='k', label='Warm Loss') 
                    plt.yscale('log')
                    plt.xlabel('Iteration') 
                    plt.ylabel('Loss (Log Scale)') 
                    plt.title('Warm Loss Over Iterations')  
                    plt.grid(True, which='both', linestyle='--', linewidth=0.5)  
                    plt.legend() 

                    # Save the plot to a file
                    plt.savefig(os.path.join(self.base_exp_dir, f"{recon_dir_name}/warmloss_{idx_so_far + i}.png"))
                    plt.clf()
                    plt.close()
            

            start_plot_lang_step = 1 + FLAGS.num_lang_steps - FLAGS.num_per_image

            fps_loss = []
            ou_terms, gd_terms, noise_terms = [], [], []
            
                
            # Parameters for exponential decay
            start_value = 1.5e-4  # Starting value
            end_value = 1.0e-4     # Ending value
            steps = FLAGS.num_lang_steps+1           # Number of steps

            # Calculate the decay rate based on the start and end values
            decay_rate = np.log(start_value / end_value) / (steps - 1)

            # Generate exponentially decaying values over the specified number of steps
            decaying_values = start_value * np.exp(-decay_rate * np.arange(steps))
            
            for _lang_step in range(FLAGS.num_lang_steps+1):
                x0 = x0.requires_grad_()
                x1_hat = self.sample_true(x0, label=label, stage="sampling")
                
                
                if FLAGS.plot_all:
                    for i in range(len(x1_hat)):
                        tvu.save_image(
                                inverse_data_transform(x1_hat[i]), os.path.join(self.base_exp_dir, f"{recon_dir_name}/samples/recon_{idx_so_far + i}_lang{_lang_step}.png")
                                )
                elif _lang_step >= start_plot_lang_step:
                    for i in range(len(x1_hat)):
                        tvu.save_image(
                                inverse_data_transform(x1_hat[i]), os.path.join(self.base_exp_dir, f"{recon_dir_name}/samples/fps_{idx_so_far + i}_repeat{_lang_step-start_plot_lang_step}.png")
                                )

                if _lang_step != FLAGS.num_lang_steps - 1:
                    L = torch.pow(torch.linalg.norm(y - self.A(x1_hat)), 2)
                    L_grad = torch.autograd.grad(outputs=L.sum(), inputs=x0)[0]
                    g = (1/(2*(FLAGS.cond_sigma**2)))*L_grad
                    
                    x0 = x0.detach_()
                    L = L.detach()
                    L_grad = L_grad.detach()
                    # g = g.detach()
                    
                    if not FLAGS.decay_tau:
                        current_tau = FLAGS.tau
                    else:
                        current_tau = decaying_values[_lang_step]
                    
                    ou_term =  -1 * current_tau*x0 / (self.sigma_max**2)
                    gd_term = -1 * current_tau*g
                    noise_term = np.sqrt(2.*current_tau)*torch.randn_like(x0)
                    
                    ou_terms.append(torch.linalg.norm(ou_term.detach().cpu()))
                    gd_terms.append(torch.linalg.norm(gd_term.detach().cpu()))
                    noise_terms.append(torch.linalg.norm(noise_term.detach().cpu()))

                    x0 = x0 + ou_term + gd_term + noise_term

                    fps_loss.append(L.detach().cpu())

            if FLAGS.plot_all:
                final_out = inverse_data_transform(x1_hat).detach()
                fps_loss = torch.stack(fps_loss, dim=0)
                for i in range(len(final_out)):
                    
                    tvu.save_image(
                            final_out[i], os.path.join(self.base_exp_dir, f"{recon_dir_name}/fps_{idx_so_far + i}.png")
                            )

                plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
                plt.plot(fps_loss, c='k', label='FPS Loss')  # Add a label for the plot line
                plt.yscale('log')
                plt.xlabel('Iteration')  # Add an x-axis label
                plt.ylabel('Loss (Log Scale)')  # Add a y-axis label
                plt.title('FPS Loss Over Iterations')  # Add a title to the plot
                plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add a grid to the plot
                plt.legend()  # Show the legend

                # Save the plot to a file
                plt.savefig(os.path.join(self.base_exp_dir, f"{recon_dir_name}/fpsloss_{idx_so_far + i}.png"))
                plt.clf()
                plt.close()

                plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
                plt.plot(ou_terms, c='r', label=r"$-\tau \cdot x_0$")
                plt.plot(gd_terms, c='k', label=r"$-\tau \cdot g$")
                plt.plot(noise_terms, c='g', label=r"$\sqrt{2\tau} \cdot \xi$")
                plt.yscale('log')
                plt.xlabel('Iteration')  # Add an x-axis label
                plt.ylabel('Value (Log Scale)')  # Add a y-axis label
                plt.title('FPS Forces Over Iterations')  # Add a title to the plot
                plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add a grid to the plot
                plt.legend()  # Show the legend

                # Save the plot to a file
                plt.savefig(os.path.join(self.base_exp_dir, f"{recon_dir_name}/fpsforces_{idx_so_far + i}.png"))
                plt.clf()
                plt.close()

            idx_so_far += y.shape[0]

            if idx_so_far >= FLAGS.num_plot + FLAGS.plot_start_index:
                break
        
        if CALCULATE_TIME:
            end_time = time.time()
            reconstruction_time = end_time - start_time
            print(f"Image reconstruction took {reconstruction_time:.4f} seconds.")
    
    def dps_for_CM(self):
        pass
    
def solve_inverse_problem(argv):
    base_exp_dir = setup_experiment_directory()
    setup_seed_and_cuda(FLAGS.seed)
    # load data config 
    with open(os.path.join(FLAGS.data_config_pth), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    inverse_flow = SolveInverseProblem(base_exp_dir, new_config, device=device)
    inverse_flow.conditional_sample()
    

if __name__ == "__main__":
    app.run(solve_inverse_problem)