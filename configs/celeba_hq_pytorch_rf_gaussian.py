# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Training rectified Flow on CelebA HQ."""

import ml_collections
import torch


def get_default_configs():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  config.training.batch_size = 64
  training.n_iters = 2400001
  training.snapshot_freq = 50000
  training.log_freq = 50
  training.eval_freq = 100
  ## store additional checkpoints for preemption in cloud computing environments
  training.snapshot_freq_for_preemption = 5000
  ## produce samples at each snapshot.
  training.snapshot_sampling = True
  training.likelihood_weighting = False
  training.continuous = True
  training.reduce_mean = False

  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.n_steps_each = 1
  sampling.noise_removal = True
  sampling.probability_flow = False
  sampling.snr = 0.075

  sampling.sigma_variance = 0.0 # NOTE: XC: sigma variance for turning ODe to SDE
  sampling.init_noise_scale = 1.0
  sampling.use_ode_sampler = 'ode'
  sampling.ode_tol = 1e-5
  sampling.sample_N = 1000

  # evaluation
  config.eval = evaluate = ml_collections.ConfigDict()
  evaluate.begin_ckpt = 50
  evaluate.end_ckpt = 96
  evaluate.batch_size = 512
  evaluate.enable_sampling = False
  evaluate.enable_figures_only = False
  evaluate.num_samples = 50000
  evaluate.enable_loss = False
  evaluate.enable_bpd = False
  evaluate.bpd_dataset = 'test'

  # data
  config.data = data = ml_collections.ConfigDict()
  data.dataset = 'LSUN'
  data.image_size = 256
  data.random_flip = True
  data.uniform_dequantization = False
  data.centered = False
  data.num_channels = 3
  data.root_path = 'YOUR_ROOT_PATH'

  # model
  config.model = model = ml_collections.ConfigDict()
  model.sigma_max = 378
  model.sigma_min = 0.01
  model.num_scales = 2000
  model.beta_min = 0.1
  model.beta_max = 20.
  model.dropout = 0.
  model.embedding_type = 'fourier'

  # optimization
  config.optim = optim = ml_collections.ConfigDict()
  optim.weight_decay = 0
  optim.optimizer = 'Adam'
  optim.lr = 2e-4
  optim.beta1 = 0.9
  optim.eps = 1e-8
  optim.warmup = 5000
  optim.grad_clip = 1.

  config.seed = 42
  config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

  return config


def get_config():
  config = get_default_configs()
  # training 
  training = config.training
  training.sde = 'rectified_flow'
  training.continuous = False
  training.reduce_mean = True
  training.snapshot_freq = 100000
  training.data_dir = 'DATA_DIR'

  # sampling
  sampling = config.sampling
  sampling.method = 'rectified_flow'
  sampling.init_type = 'gaussian' 
  sampling.init_noise_scale = 1.0
  sampling.use_ode_sampler = 'rk45'

  # data
  data = config.data
  data.dataset = 'CelebA-HQ-Pytorch'
  data.centered = True

  # model
  model = config.model
  model.name = 'ncsnpp'
  model.scale_by_sigma = True
  model.ema_rate = 0.999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 128
  model.ch_mult = (1, 1, 2, 2, 2, 2, 2)
  model.num_res_blocks = 2
  model.attn_resolutions = (16,)
  model.resamp_with_conv = True
  model.conditional = True
  model.fir = True
  model.fir_kernel = [1, 3, 3, 1]
  model.skip_rescale = True
  model.resblock_type = 'biggan'
  model.progressive = 'output_skip'
  model.progressive_input = 'input_skip'
  model.progressive_combine = 'sum'
  model.attention_type = 'ddpm'
  model.init_scale = 0.
  model.fourier_scale = 16
  model.conv_size = 3

  return config