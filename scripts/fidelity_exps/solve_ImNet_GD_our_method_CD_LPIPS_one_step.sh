#!/bin/bash
export LD_LIBRARY_PATH=/depot/qqiu/data/vishal/envs/compressive/lib/:$LD_LIBRARY_PATH
SCRIPT="solve_inverse_problem_flowcm.py"

# Data & Exp related args
DATASET="lsun_bedroom"
EXP_DIR="./inverse_problem_results_our_method"
DATA_DIR="/depot/qqiu/data/vishal/03_Flow_Posterior_Sampling/flowinverse_matt/00_datasets/lsun" 
DATA_CONFIG="./configs/lsun_bedroom.yaml"

# Backbone related args
MODEL="cm"
CKPT_PTH="/depot/qqiu/data/vishal/03_Flow_Posterior_Sampling/flowinverse_matt/00_pre_trained_models/CM_models/cd_bedroom256_lpips.pt"

# Degradation args
DEG="gaussian_deblur"
DEG_SCALE=4.0 # this does nothing in denoising case # for inpainting and SR use
SIGMA=0.1 # the true noise level of inverse problem
MASK_PTH="/depot/qqiu/data/vishal/03_Flow_Posterior_Sampling/flowinverse_matt/00_masks"

# Sampling related args
SAMPLING_METHOD="fps" # can also use dflow for lbfgs optimization
SDE_SOLVER="EM" # can be EM, EI, or GD
COND_SIGMA=0.1 # must be positive, typically should be the same as SIGMA
TAU=1.0e-6 # does not matter when we are decaying the tau value in prefixed range
BLEND_ALPHA=0.45 # blend init hyper-parameter
INIT_TYPE="warm" # can be rand, Apy, or gt
WARM_INIT_TYPE="rand"
WARM_SOLVER="ADAM"
WARM_LR=0.005 # not applicable for SAM warmup
WARM_MOMENTUM=0.9
WARM_STEPS=800
NUM_LANG_STEPS=10 
ODE_GRAD=cm # use consistency model to get the gradients
CM_SOLVER="onestep" # supported solvers - "heun", "dpm", "ancestral", "onestep", "progdist", "euler", "multistep"
NUM_DIFFUSION_STEPS=40 # this is useful for multi-step solver.
NUM_CM_STEPS=5 # (only for multistep solvers) there is limit as to how many steps we need to use depending on inverse problem 

# Regularizer related args 
REG="none"
REG_LAM=0.01

# book-keeping
NUM_PLOT=18 # number of samples to plot 
PLOT_ALL=True
START_INDEX=83

python $SCRIPT \
  --model=$MODEL \
  --dataset=$DATASET \
  --exp_dir=$EXP_DIR \
  --data_dir=$DATA_DIR \
  --ckpt_path=$CKPT_PTH \
  --data_config_pth=$DATA_CONFIG \
  --deg=$DEG \
  --deg_scale=$DEG_SCALE \
  --sigma=$SIGMA \
  --mask_path=$MASK_PTH \
  --sampling_method=$SAMPLING_METHOD \
  --num_plot=$NUM_PLOT \
  --init_type=$INIT_TYPE \
  --warm_init_type=$WARM_INIT_TYPE \
  --blend_alpha=$BLEND_ALPHA \
  --warm_lr=$WARM_LR \
  --warm_solver=$WARM_SOLVER \
  --num_warm_steps=$WARM_STEPS \
  --warm_momentum=$WARM_MOMENTUM \
  --cm_solver=$CM_SOLVER \
  --num_diffusion_steps=$NUM_DIFFUSION_STEPS \
  --numsteps_cm=$NUM_CM_STEPS \
  --sde_solver=$SDE_SOLVER \
  --tau=$TAU \
  --user_exp_suffix=$USR_EXP_NAME \
  --cond_sigma=$COND_SIGMA \
  --num_lang_steps=$NUM_LANG_STEPS \
  --regularizer=$REG \
  --reg_lam=$REG_LAM \
  --plot_start_index=$START_INDEX \
  --plot_all=$PLOT_ALL \
  --verbose \
