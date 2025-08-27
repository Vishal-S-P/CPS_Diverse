import copy
import os
import yaml
import random 
import tqdm
import argparse
import collections
import torch
import torch.nn.functional as F
from absl import app, flags
import torchvision
import torchvision.utils as tvu
import matplotlib.pyplot as plt

from sampler import karras_sample, KarrasDenoiser, get_generator

from models.unet.unet_lsun import UNetModel
from models.unet.unet_act import UNet2DModel
from load_edm_model import load_edm_model
from img_datasets.image_datasets import get_val_dataset, data_transform, inverse_data_transform
import numpy as np
from degradations.operators import color2gray, gray2color, MeanUpsample, MotionBlurOperator, GaussialBlurOperator, NonlinearBlurOperator
from degradations.phase_retrieval import fft2_m, ifft2_m
import torch.utils.data as data

from ms_ssim_loss import MS_SSIM_L1_LOSS
import time
CALCULATE_TIME=False
FLAGS = flags.FLAGS
flags.DEFINE_integer("num_workers", 1, help="workers of Dataloader")

# Model related args
flags.DEFINE_string("model", "fm", help="flow matching model type")
flags.DEFINE_string("ckpt_path", "./", help="ckpt_dir")
flags.DEFINE_string("distil_ckpt_path", "./", help="ckpt_dir")
flags.DEFINE_string("exp_dir", "./inverse_problem_results", help="results_dir for inverse problem results")
flags.DEFINE_bool("load_ema", True, help="Loads EMA ckpt")

# Dataset & Dataloader
flags.DEFINE_string("dataset", "afhq", help="Datset for inverse problem solving")
flags.DEFINE_string("data_dir", "./data/afhq_v2/test", help="Test datset path for inverse problem solving")
flags.DEFINE_string("data_config_pth", "./configs/afhq.yml", help="yml file for data config")
flags.DEFINE_integer("batch_size", 1, help="batch size")
flags.DEFINE_bool("class_cond", False, help="Class conditioning") # TODO: not supported yet

# Inverse problem related args
flags.DEFINE_integer("seed", 44, help="seed")
flags.DEFINE_string("deg", "inpaint", help="Degradation selector")
flags.DEFINE_float("deg_scale", 0.1, help="degradation scale")
flags.DEFINE_float("sigma", 0.05, help="noise level") 
flags.DEFINE_string("mask_path", "./", help="path for obtaining pre-determined masks")

# Sampling related args
flags.DEFINE_string("sampling_method", "dpscm", help="Supported methods include dpscm | cminv | cmedit | cmdps | dps | mpgd | lgd")

# DPS-CM args
flags.DEFINE_integer("num_diffusion_steps", 40, help="Number of diffusion steps")
flags.DEFINE_float("zeta", 7., help="weight of CM correction")

flags.DEFINE_integer("mcsamples", 5, help="Number of samples (inner loop) for loss guided diffusion")

# CM Inversion Args
flags.DEFINE_float("zeta1", 1e-1, help="init noise learning rate")
flags.DEFINE_float("zeta2", 1e-3, help="main body noise learning rate")
flags.DEFINE_integer("numsteps_cm", 5, help="Number of sampling steps for cm (outer loop)")
flags.DEFINE_integer("noise_opt_steps", 151, help="number of noise optimization steps (inner loop)")

flags.DEFINE_integer("num_plot", 5, help="number of recons to plot")

flags.DEFINE_integer("num_per_image", 1, help="number of recons to get for each image")

flags.DEFINE_bool("verbose", True, help="Show tqdm progress bar") 
flags.DEFINE_bool("use_CM", False, help="Which backbone to use while running baseline methods.") 
flags.DEFINE_integer("plot_start_index", 0, help="Index of validation set")
flags.DEFINE_bool("plot_all", False, help="plot all intermediate samples of diffusion") 

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

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

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]

class OTFlow(object):
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
        self.distil_ckpt_path = FLAGS.distil_ckpt_path
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
            if self.degradation == "random_mask_inpaint":
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


        elif self.degradation == "interactive_colorization":
            # hint-based or user-guided colorization task 
            pass

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
            
        elif self.degradation == "motion_deblur":
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
        
        
        # Model setup and ckpt loading 
        if FLAGS.dataset == "lsun_bedroom" or FLAGS.dataset == "lsun_cat":
            if FLAGS.sampling_method in [ "dps", "mpgd", "lgd"] and not FLAGS.use_CM : 
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
                print("[INFO] Using Diffusion Model Backbone....")
                if self.ckpt_path is not None:
                    checkpoint = torch.load(self.ckpt_path, map_location=self.device)
                    
                    self.net_model.load_state_dict(checkpoint)
                    self.net_model.to(self.device)
                    
                    self.net_model.convert_to_fp16()
                    print(f"Model loaded from checkpoint: {self.ckpt_path}")
                else:
                    raise FileNotFoundError("Checkpoint path incorrect!!")

            elif FLAGS.sampling_method in ["dps", "cminv", "cmedit", "mpgd", "lgd"] and FLAGS.use_CM:
                self.distil_rescale_t = True
                self.distiller = UNetModel(
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
                print("[INFO] Using Consistency Model Backbone....")
                if self.distil_ckpt_path is not None:
                    checkpoint = torch.load(self.distil_ckpt_path, map_location=self.device)
                    self.distiller.load_state_dict(checkpoint)
                    self.distiller.to(self.device)
                    self.distiller.convert_to_fp16()
                else:
                    raise FileNotFoundError("Distil Checkpoint path incorrect!!")

            else:
                raise NotImplementedError("Sampler type no supported!!")
            
            # parameters from Karras et al. "Elucidating the Design Space of Diffusion Models."
            self.sigma_data = 0.5
            self.sigma_max = 80.0
            self.sigma_min = 0.002

            self.steps = FLAGS.num_diffusion_steps

            ramp = torch.linspace(0, 1, self.steps)
            min_inv_rho = self.sigma_min ** (1 / 7.0)
            max_inv_rho = self.sigma_max ** (1 / 7.0)
            sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** 7.0
            self.sigmas = torch.cat([sigmas, sigmas.new_zeros([1])]).to(self.device)

            self.clip_denoised = True

            self.zeta = FLAGS.zeta

            self.generator = get_generator("determ", 1, 42)

            if FLAGS.sampling_method in ["dps", "cminv", "cmedit", "mpgd", "lgd"] and FLAGS.use_CM:
                self.diffusion_distil = KarrasDenoiser(
                        sigma_data=self.sigma_data,
                        sigma_max=self.sigma_max,
                        sigma_min=self.sigma_min,
                        distillation=True,
                        weight_schedule="uniform",
                )
            elif FLAGS.sampling_method in [ "dps", "mpgd", "lgd"] and not FLAGS.use_CM : 
                self.diffusion = KarrasDenoiser(
                        sigma_data=self.sigma_data,
                        sigma_max=self.sigma_max,
                        sigma_min=self.sigma_min,
                        distillation=False,
                        weight_schedule="uniform",
                )
            
            
        else:
            raise NotImplementedError("Dataset not supported...")
        
    
    def denoiser(self, x_t, sigma, label=None):
        model_out, denoised = self.diffusion.denoise(self.net_model, x_t, sigma, t_rescale=self.rescale_t)
        if self.clip_denoised:
            denoised = denoised.clamp(-1, 1)
        return model_out, denoised

    def denoiserdistiller(self, x_t, sigma, label=None):
        _, denoised = self.diffusion_distil.denoise(self.distiller, x_t, sigma, t_rescale=self.distil_rescale_t)
        if self.clip_denoised:
            denoised = denoised.clamp(-1, 1)
        return denoised

    def get_ancestral_step(self, sigma_from, sigma_to):
        """Calculates the noise level (sigma_down) to step down to and the amount
        of noise to add (sigma_up) when doing an ancestral sampling step."""
        sigma_up = (
            sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2
        ) ** 0.5
        sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
        return sigma_down, sigma_up


    def conditional_sample(self):
        """
        Dataset, degradation and sampling
        """
        if FLAGS.sampling_method == "cminv":
            self.cm_inversion()
        elif FLAGS.sampling_method == "cmedit":
            self.cm_edit()
        elif FLAGS.sampling_method == "dps":
            self.diffusion_posterior_sampling()
        elif FLAGS.sampling_method == "mpgd":
            self.manifold_preserving_guided_diffusion()
        elif FLAGS.sampling_method == "lgd":
            self.loss_guided_diffusion()
        else:
            raise NotImplementedError("Sampler not found...")


    def cm_inversion(self):
        idx_so_far = 0

        if FLAGS.batch_size!=1:
            raise ValueError("please set batch size to 1 for inverse problems")


        recon_dir_name = "cminv_"
        recon_dir_name += "zeta1_{:.1e}_".format(FLAGS.zeta1)
        recon_dir_name += "zeta2_{:.1e}_".format(FLAGS.zeta2)
        recon_dir_name += "noiseopt{}_".format(FLAGS.noise_opt_steps)
        recon_dir_name += "cmsteps{}_".format(FLAGS.numsteps_cm)
        recon_dir_name += "steps{}".format(self.steps)

        print(recon_dir_name)
        os.makedirs(os.path.join(self.base_exp_dir, recon_dir_name), exist_ok=True)
        export_flags_to_yaml(os.path.join(os.path.join(self.base_exp_dir, recon_dir_name), "flags.yaml"))


        pbar = tqdm.tqdm(self.val_loader) if FLAGS.verbose else self.val_loader
        for x_orig, label in pbar:

            
            if idx_so_far < FLAGS.plot_start_index:
                idx_so_far += 1
                if idx_so_far >= FLAGS.num_plot + FLAGS.plot_start_index:
                    break
                else:
                    continue


            x_orig = x_orig.to(self.device)
            x_orig = data_transform(x_orig) 

            y = self.A(x_orig).detach()
            y += torch.randn_like(y)*FLAGS.sigma

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

            shape = x_orig.shape

            for k in range(FLAGS.num_per_image):
                x0 = self.generator.randn(*shape, device=self.device).detach() * self.sigma_max

                if os.path.exists(os.path.join(self.base_exp_dir, f"{recon_dir_name}/cminv_{idx_so_far}_repeat{k}.png")):
                    continue

                s_in = x0.new_ones([x0.shape[0]]) 
                t_max_rho = self.sigma_max ** (1 / 7.0)
                t_min_rho = self.sigma_min ** (1 / 7.0)

                ts=list(np.linspace(0,self.steps-1,FLAGS.numsteps_cm,dtype=int))

                x0 = x0.detach().clone().requires_grad_()
                noise_optimizer = torch.optim.Adam([x0], lr=FLAGS.zeta1)

                for opt_step in range(FLAGS.noise_opt_steps):
                    _x0 = self.denoiserdistiller(x0, self.sigmas[0] * s_in, label.to(self.device))
                    
                    difference = y - self.A(_x0)
                    diffnorm = torch.linalg.norm(difference)
                    diffnorm.backward()
                    
                    noise_optimizer.step()
                    noise_optimizer.zero_grad()
                x0 = _x0.detach()
                
                for i in range(1,len(ts)-1):

                    t = (t_max_rho + ts[i] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** 7.0

                    cm_noise = self.generator.randn_like(x0)
                    cm_noise = cm_noise.requires_grad_()
                    noise_optimizer = torch.optim.Adam([cm_noise], lr=FLAGS.zeta2)
                    
                    for opt_step in range(FLAGS.noise_opt_steps):
                        x = x0.detach().clone() + cm_noise * np.sqrt(t**2 - self.sigma_min**2)
                        _x0 = self.denoiserdistiller(x, t * s_in, label.to(self.device))
                        
                        difference = y - self.A(_x0)
                        diffnorm = torch.linalg.norm(difference)
                        diffnorm.backward()
                        
                        noise_optimizer.step()
                        noise_optimizer.zero_grad()
                    x0 = _x0.detach()

                final_out = inverse_data_transform(x0).detach()

                for j in range(len(final_out)):    
                    tvu.save_image(
                        final_out[j], os.path.join(self.base_exp_dir, f"{recon_dir_name}/cminv_{idx_so_far}_repeat{k}.png")
                        )


            idx_so_far += y.shape[0]

            if idx_so_far >= FLAGS.num_plot + FLAGS.plot_start_index:
                break


    def diffusion_posterior_sampling(self):
        idx_so_far = 0

        if FLAGS.batch_size!=1:
            raise ValueError("please set batch size to 1 for inverse problems")


        recon_dir_name = "dps_"
        recon_dir_name += "zeta{:.1e}_".format(self.zeta)
        recon_dir_name += "steps{}".format(self.steps)
        if FLAGS.use_CM:
            recon_dir_name += "cm_backbone"
        else:
            recon_dir_name += "dm_backbone"
        print(recon_dir_name)
        os.makedirs(os.path.join(self.base_exp_dir, recon_dir_name), exist_ok=True)
        export_flags_to_yaml(os.path.join(os.path.join(self.base_exp_dir, recon_dir_name), "flags.yaml"))


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

            label = label.to(self.device)

            x_orig = x_orig.to(self.device)
            x_orig = data_transform(x_orig) 

            if self.degradation == "avg_patch_downsampling":
                y = self.A(x_orig, 256, round(FLAGS.deg_scale))
            else:
                y = self.A(x_orig).detach()
            y += torch.randn_like(y)*FLAGS.sigma

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

            shape = x_orig.shape

            for k in range(FLAGS.num_per_image):
                x0 = self.generator.randn(*shape, device=self.device).detach() * self.sigma_max

                if os.path.exists(os.path.join(self.base_exp_dir, f"{recon_dir_name}/dps_{idx_so_far}_repeat{k}.png")):
                    continue

                s_in = x0.new_ones([x0.shape[0]]) 
                steps = len(self.sigmas)
                indices = range(steps - 1)
                for i in indices:

                    with torch.enable_grad():
                        x_ = x0.detach().clone().requires_grad_()
                        if FLAGS.use_CM:
                            denoised = self.denoiserdistiller(x_, self.sigmas[i] * s_in, label=label)
                        else:
                            model_out, denoised = self.denoiser(x_, self.sigmas[i] * s_in, label=label)
                            
                        if self.degradation == "avg_patch_downsampling":
                            difference = y - self.A(denoised, 256, round(FLAGS.deg_scale))
                        else:
                            difference = y - self.A(denoised)
                        norm = torch.linalg.norm(difference)
                        
                        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_)[0]
                    

                    sigma_down, sigma_up = self.get_ancestral_step(self.sigmas[i], self.sigmas[i + 1])
                    
                    d = (x0 - denoised) / append_dims(self.sigmas[i], x0.ndim)
                    dt = sigma_down - self.sigmas[i]
                    x0 = x0 + d * dt
                    x0 = x0 + self.generator.randn_like(x0) * sigma_up
                    offset = self.zeta * norm_grad * self.sigmas[i]
                    x0 = x0 - offset
                    x0 = x0.detach()


                final_out = inverse_data_transform(x0).detach()

                for j in range(len(final_out)):    
                    tvu.save_image(
                        final_out[j], os.path.join(self.base_exp_dir, f"{recon_dir_name}/dps_{idx_so_far}_repeat{k}.png")
                        )


            idx_so_far += y.shape[0]

            if idx_so_far >= FLAGS.num_plot + FLAGS.plot_start_index:
                break
            
        if CALCULATE_TIME:
            end_time = time.time()
            reconstruction_time = end_time - start_time
            print(f"Image reconstruction took {reconstruction_time:.4f} seconds.")

    def manifold_preserving_guided_diffusion(self):
        idx_so_far = 0

        if FLAGS.batch_size!=1:
            raise ValueError("please set batch size to 1 for inverse problems")


        recon_dir_name = "mpgd_"
        recon_dir_name += "zeta{:.1e}_".format(self.zeta)
        recon_dir_name += "steps{}".format(self.steps)
        if FLAGS.use_CM:
            recon_dir_name += "cm_backbone"
        else:
            recon_dir_name += "dm_backbone"
            
        print(recon_dir_name)
        os.makedirs(os.path.join(self.base_exp_dir, recon_dir_name), exist_ok=True)
        export_flags_to_yaml(os.path.join(os.path.join(self.base_exp_dir, recon_dir_name), "flags.yaml"))


        pbar = tqdm.tqdm(self.val_loader) if FLAGS.verbose else self.val_loader
        for x_orig, label in pbar:

            
            if idx_so_far < FLAGS.plot_start_index:
                idx_so_far += 1
                if idx_so_far >= FLAGS.num_plot + FLAGS.plot_start_index:
                    break
                else:
                    continue


            x_orig = x_orig.to(self.device)
            x_orig = data_transform(x_orig) 

            if self.degradation == "avg_patch_downsampling":
                y = self.A(x_orig, 256, round(FLAGS.deg_scale)).detach()
            else:
                y = self.A(x_orig).detach()
            y += torch.randn_like(y)*FLAGS.sigma

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

            shape = x_orig.shape

            for k in range(FLAGS.num_per_image):
                x0 = self.generator.randn(*shape, device=self.device).detach() * self.sigma_max

                if os.path.exists(os.path.join(self.base_exp_dir, f"{recon_dir_name}/mpgd_{idx_so_far}_repeat{k}.png")):
                    continue

                s_in = x0.new_ones([x0.shape[0]]) 
                steps = len(self.sigmas)
                indices = range(steps - 1)
                for i in indices:
                    x_ = x0.detach().clone().requires_grad_()
                    if FLAGS.use_CM:
                        denoised = self.denoiserdistiller(x_, self.sigmas[i] * s_in, label=label)
                    else:
                        _, denoised = self.denoiser(x_, self.sigmas[i] * s_in, label=label)
                    with torch.enable_grad():
                        denoised_ = denoised.detach().clone().requires_grad_()
                        # denoised_ = denoised
                        if self.degradation == "avg_patch_downsampling":
                            difference = y - self.A(denoised_, 256, round(FLAGS.deg_scale))
                        else:
                            difference = y - self.A(denoised_)
                        
                        norm = torch.linalg.norm(difference)
                        
                        norm_grad = torch.autograd.grad(outputs=norm, inputs=denoised_)[0]
                    
                    offset = self.zeta * norm_grad * 10
                    sigma_down, sigma_up = self.get_ancestral_step(self.sigmas[i], self.sigmas[i + 1])

                    denoised -= offset
                    d = (x0 - denoised) / append_dims(self.sigmas[i], x0.ndim)
                    dt = sigma_down - self.sigmas[i]
                    x0 = x0 + d * dt
                    x0 = x0 + self.generator.randn_like(x0) * sigma_up
                    x0 = x0.detach()


                final_out = inverse_data_transform(x0).detach()

                for j in range(len(final_out)):    
                    tvu.save_image(
                        final_out[j], os.path.join(self.base_exp_dir, f"{recon_dir_name}/mpgd_{idx_so_far}_repeat{k}.png")
                        )


            idx_so_far += y.shape[0]

            if idx_so_far >= FLAGS.num_plot + FLAGS.plot_start_index:
                break


    def loss_guided_diffusion(self):
        idx_so_far = 0

        if FLAGS.batch_size!=1:
            raise ValueError("please set batch size to 1 for inverse problems")


        recon_dir_name = "lgd_"
        recon_dir_name += "zeta{:.1e}_".format(self.zeta)
        recon_dir_name += "mcsamples{}_".format(FLAGS.mcsamples)
        recon_dir_name += "steps{}".format(self.steps)
        if FLAGS.use_CM:
            recon_dir_name += "cm_backbone"
        else:
            recon_dir_name += "dm_backbone"
            
        print(recon_dir_name)
        os.makedirs(os.path.join(self.base_exp_dir, recon_dir_name), exist_ok=True)
        export_flags_to_yaml(os.path.join(os.path.join(self.base_exp_dir, recon_dir_name), "flags.yaml"))


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


            x_orig = x_orig.to(self.device)
            x_orig = data_transform(x_orig) 

            if self.degradation == "avg_patch_downsampling":
                y = self.A(x_orig, 256, round(FLAGS.deg_scale)).detach()
            else:
                y = self.A(x_orig).detach()
            y += torch.randn_like(y)*FLAGS.sigma

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

            shape = x_orig.shape

            for k in range(FLAGS.num_per_image):
                x0 = self.generator.randn(*shape, device=self.device).detach() * self.sigma_max

                if os.path.exists(os.path.join(self.base_exp_dir, f"{recon_dir_name}/lgd_{idx_so_far}_repeat{k}.png")):
                    continue

                s_in = x0.new_ones([x0.shape[0]]) 
                steps = len(self.sigmas)
                indices = range(steps - 1)
                for i in indices:

                    mcnoise = self.sigmas[i] / torch.sqrt(1 + torch.pow(self.sigmas[i],2))
                    
                    with torch.enable_grad():
                        x_ = x0.detach().clone().requires_grad_()
                        if FLAGS.use_CM:
                            denoised = self.denoiserdistiller(x_, self.sigmas[i] * s_in, label=label)
                        else:
                            model_out, denoised = self.denoiser(x_, self.sigmas[i] * s_in, label=label)
                        
                        if self.degradation == "avg_patch_downsampling":
                            difference = y - self.A(denoised+ torch.randn_like(denoised) * mcnoise, 256, round(FLAGS.deg_scale))
                        else:
                            difference = y - self.A(denoised + torch.randn_like(denoised) * mcnoise)
                        
                        norm = torch.linalg.norm(difference)
                        for _ in range(FLAGS.mcsamples - 1):
                            if self.degradation == "avg_patch_downsampling":
                                difference = y - self.A(denoised+ torch.randn_like(denoised) * mcnoise, 256, round(FLAGS.deg_scale))
                            else:
                                difference = y - self.A(denoised + torch.randn_like(denoised) * mcnoise)
                            norm = norm + torch.linalg.norm(difference)
                        
                        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_)[0]
                    

                    sigma_down, sigma_up = self.get_ancestral_step(self.sigmas[i], self.sigmas[i + 1])
                    
                    d = (x0 - denoised) / append_dims(self.sigmas[i], x0.ndim)
                    dt = sigma_down - self.sigmas[i]
                    x0 = x0 + d * dt
                    x0 = x0 + self.generator.randn_like(x0) * sigma_up
                    offset = self.zeta * norm_grad * self.sigmas[i]
                    x0 = x0 - offset
                    x0 = x0.detach()


                final_out = inverse_data_transform(x0).detach()

                for j in range(len(final_out)):    
                    tvu.save_image(
                        final_out[j], os.path.join(self.base_exp_dir, f"{recon_dir_name}/lgd_{idx_so_far}_repeat{k}.png")
                        )


            idx_so_far += y.shape[0]

            if idx_so_far >= FLAGS.num_plot + FLAGS.plot_start_index:
                break
        
        if CALCULATE_TIME:
            end_time = time.time()
            reconstruction_time = end_time - start_time
            print(f"Image reconstruction took {reconstruction_time:.4f} seconds.")

    @torch.no_grad()
    def cm_edit(self):
        idx_so_far = 0

        if FLAGS.batch_size!=1:
            raise ValueError("please set batch size to 1 for inverse problems")


        recon_dir_name = "cmedit_"
        recon_dir_name += "cmsteps{}_".format(FLAGS.numsteps_cm)
        recon_dir_name += "steps{}".format(self.steps)

        if self.degradation == "sr_avgpooling":
            
            patch_size = round(self.deg_scale)
            
            def average_image_patches(x, image_size):
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
                
            def obtain_orthogonal_matrix():
                vector = np.asarray([1] * patch_size**2)
                vector = vector / np.linalg.norm(vector)
                matrix = np.eye(patch_size**2)
                matrix[:, 0] = vector
                matrix = np.linalg.qr(matrix)[0]
                if np.sum(matrix[:, 0]) < 0:
                    matrix = -matrix
                return matrix

            

            def replacement(x0, x1, image_size):
                x0_flatten = (
                    x0.reshape(-1, 3, image_size, image_size)
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
                x1_flatten = (
                    x1.reshape(-1, 3, image_size, image_size)
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
                x0 = torch.einsum("bcnd,de->bcne", x0_flatten, Q)
                x1 = torch.einsum("bcnd,de->bcne", x1_flatten, Q)
                x_mix = x0.new_zeros(x0.shape)
                x_mix[..., 0] = x0[..., 0]
                x_mix[..., 1:] = x1[..., 1:]
                x_mix = torch.einsum("bcne,de->bcnd", x_mix, Q)
                x_mix = (
                    x_mix.reshape(
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
                return x_mix
            
            Q = torch.from_numpy(obtain_orthogonal_matrix()).to(self.device).to(torch.float32)
        elif self.degradation == "random_pixel_inpaint":
            loaded = np.load(os.path.join(FLAGS.mask_path, "random_pixel", f"{self.deg_scale}_percent", "mask.npy")).astype(np.float32)
            mask = torch.from_numpy(loaded).to(self.device)
            
            def replacement(x0, x1):
                x_mix = x0 * mask + x1 * (1 - mask)
                return x_mix
        else:
            raise NotImplementedError

        print(recon_dir_name)
        os.makedirs(os.path.join(self.base_exp_dir, recon_dir_name), exist_ok=True)
        export_flags_to_yaml(os.path.join(os.path.join(self.base_exp_dir, recon_dir_name), "flags.yaml"))


        pbar = tqdm.tqdm(self.val_loader) if FLAGS.verbose else self.val_loader
        for x_orig, label in pbar:
            if idx_so_far < FLAGS.plot_start_index:
                idx_so_far += 1
                if idx_so_far >= FLAGS.num_plot + FLAGS.plot_start_index:
                    break
                else:
                    continue


            x_orig = x_orig.to(self.device)
            x_orig = data_transform(x_orig) 

            if self.degradation == "sr_avgpooling":
                y = average_image_patches(x_orig, 256)
            elif self.degradation == "random_pixel_inpaint":
                y = replacement(x_orig, -torch.ones_like(x_orig))
            else:
                raise NotImplementedError
            y += torch.randn_like(y)*FLAGS.sigma
            
            for i in range(len(y)):
                tvu.save_image(
                    inverse_data_transform(y[i]),
                    os.path.join(self.base_exp_dir, f"y/y_{idx_so_far + i}.png")
                )
                tvu.save_image(
                    inverse_data_transform(x_orig[i]),
                    os.path.join(self.base_exp_dir, f"gt/orig_{idx_so_far + i}.png")
                )

            shape = x_orig.shape
            image_size = shape[-1]

            for k in range(FLAGS.num_per_image):
                x0 = self.generator.randn(*shape, device=self.device).detach() * self.sigma_max

                if os.path.exists(os.path.join(self.base_exp_dir, f"{recon_dir_name}/cmedit_{idx_so_far}_repeat{k}.png")):
                    continue

                s_in = x0.new_ones([x0.shape[0]]) 
                t_max_rho = self.sigma_max ** (1 / 7.0)
                t_min_rho = self.sigma_min ** (1 / 7.0)

                ts=list(np.linspace(0,self.steps-1,FLAGS.numsteps_cm,dtype=int))

                for i in range(len(ts)-1):

                    t = (t_max_rho + ts[i] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** 7.0
                    _x0 = self.denoiserdistiller(x0, t * s_in, label.to(self.device))
                    _x0 = torch.clamp(_x0, -1.0, 1.0)
                    if self.degradation == "sr_avgpooling":
                        _x0 = replacement(y, _x0, image_size)
                    elif self.degradation == "random_pixel_inpaint":
                        _x0 = replacement(y, _x0)
                    next_t = (t_max_rho + ts[i + 1] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** 7.0
                    next_t = np.clip(next_t, self.sigma_min, self.sigma_max)
                    x0 = _x0 + self.generator.randn_like(x0) * np.sqrt(next_t**2 - self.sigma_min**2)


                final_out = inverse_data_transform(x0).detach()

                for j in range(len(final_out)):    
                    tvu.save_image(
                        final_out[j], os.path.join(self.base_exp_dir, f"{recon_dir_name}/cmedit_{idx_so_far}_repeat{k}.png")
                        )


            idx_so_far += y.shape[0]

            if idx_so_far >= FLAGS.num_plot + FLAGS.plot_start_index:
                break


def solve_inverse_problem(argv):
    base_exp_dir = setup_experiment_directory()
    setup_seed_and_cuda(FLAGS.seed)

    # load data config 
    with open(os.path.join(FLAGS.data_config_pth), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    if FLAGS.model == "cm":
        inverse_flow = OTFlow(base_exp_dir, new_config, device=device)
        inverse_flow.conditional_sample()
    else:
        raise NotImplementedError

if __name__ == "__main__":
    app.run(solve_inverse_problem)