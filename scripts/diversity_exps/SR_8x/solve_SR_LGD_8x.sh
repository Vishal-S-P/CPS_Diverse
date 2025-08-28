#!/bin/bash
export LD_LIBRARY_PATH=/depot/qqiu/data/vishal/envs/compressive/lib/:$LD_LIBRARY_PATH
SCRIPT="solve_inverse_problem_baselines.py"

# Data & Exp related args
DATASET="lsun_bedroom"
EXP_DIR="./inverse_problem_results_baseline_diversity/"
DATA_DIR="/depot/qqiu/data/vishal/03_Flow_Posterior_Sampling/flowinverse_matt/00_datasets/lsun" 
DATA_CONFIG="./configs/lsun_bedroom.yaml"

# Backbone related args
MODEL="cm"
# Only usef for Diffusion based sampling methods
CKPT_PTH="/depot/qqiu/data/vishal/03_Flow_Posterior_Sampling/flowinverse_matt/00_pre_trained_models/EDM_models/edm_bedroom256_ema.pt"
# Only used for CM based sampling methods
DISTILL_CKPT_PTH="/depot/qqiu/data/vishal/03_Flow_Posterior_Sampling/flowinverse_matt/00_pre_trained_models/CM_models/cd_bedroom256_lpips.pt"

# Degradation args
# DEG="avg_patch_downsampling"
DEG="sr_avgpooling"
DEG_SCALE=8.0 # this does nothing in denoising case # for inpainting and SR use
SIGMA=0.1 # the true noise level of inverse problem
MASK_PTH="/depot/qqiu/data/vishal/03_Flow_Posterior_Sampling/flowinverse_matt/00_masks"


# Sampling related arguments
SAMPLING_METHOD="lgd"  # supported methods - ["cminv", "cmedit", "dps", "mpgd", "lgd"]
USE_CM=False
NUM_DIFFUSION_STEPS=100 
NUM_CM_STEPS=40 
NUM_POSTERIOR_SAMPLES=25

## DPS ## MPGD ## LGD
ZETA=15.0

## LGD
MCSAMPLES=1

## CM-INVERSION
ZETA1=1e-1
ZETA2=1e-3
NOISE_OPTIM_STEPS=25

# Regularizer related args 
REG="none"
REG_LAM=0.01

# book-keeping
NUM_PLOT=100 # number of samples to plot 
START_INDEX=64
PLOT_ALL=True

python $SCRIPT \
  --model=$MODEL \
  --dataset=$DATASET \
  --exp_dir=$EXP_DIR \
  --data_dir=$DATA_DIR \
  --ckpt_path=$CKPT_PTH \
  --distil_ckpt_path=$DISTILL_CKPT_PTH \
  --data_config_pth=$DATA_CONFIG \
  --deg=$DEG \
  --deg_scale=$DEG_SCALE \
  --sigma=$SIGMA \
  --mask_path=$MASK_PTH \
  --sampling_method=$SAMPLING_METHOD \
  --num_plot=$NUM_PLOT \
  --zeta=$ZETA \
  --mcsamples=$MCSAMPLES \
  --num_diffusion_steps=$NUM_DIFFUSION_STEPS \
  --numsteps_cm=$NUM_CM_STEPS \
  --zeta1=$ZETA1 \
  --zeta2=$ZETA2 \
  --noise_opt_steps=$NOISE_OPTIM_STEPS \
  --num_per_image=$NUM_POSTERIOR_SAMPLES \
  --plot_all=$PLOT_ALL \
  --plot_start_index=$START_INDEX \
  --use_CM=$USE_CM \
  --verbose \
