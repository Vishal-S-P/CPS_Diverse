# [CVPR 2025] Consistency Posterior Sampling for Diverse Image Synthesis
This is the official PyTorch implementation of Consistency Posterior Sampling for Diverse Image Synthesis [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Purohit_Consistency_Posterior_Sampling_for_Diverse_Image_Synthesis_CVPR_2025_paper.pdf)

![Alt text](/contents/poster.png?raw=trues)

## 🛠 Setup
```bash
conda create -n cps_diverse python=3.11 -y
conda activate cps_diverse
pip install -r requirements.txt
```
Download pretrained diffusion and consistency models from following links and place them in 'pretrained_models' folder -
| Dataset | Diffusion Model | Consistency Model |
| :------- |:--------:| --------:|
| LSUN-Bedroom     | [LSUN-Bedroom-DM](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/lsun_bedroom.pt)  |  [LSUN-Bedroom-CM](https://openaipublic.blob.core.windows.net/consistency/cd_bedroom256_l2.pt)   |
| ImageNet   |  [ImageNet-DM](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt)  |  [ImageNet-CM](https://openaipublic.blob.core.windows.net/consistency/cd_imagenet64_l2.pt)   |

Download datasets from [OpenAI repo](https://github.com/openai/improved-diffusion/tree/main/datasets) and place them in 'sample_dataset' folder.
 

## ⚙️ Fidelity Experiments
```bash
#------------------------------------ LSUN-Beadroom (256 x 256) ------------------------------------

## 8x Super-Resolution
CPS_Diverse/scripts/fidelity_exps/solve_SR_our_method_CD_LPIPS_8x_one_step.sh
CPS_Diverse/scripts/fidelity_exps/solve_SR_our_method_CD_LPIPS_8x_multi_step.sh

## Gaussian Deblur
CPS_Diverse/scripts/fidelity_exps/solve_GD_our_method_CD_LPIPS_one_step.sh
CPS_Diverse/scripts/fidelity_exps/solve_GD_our_method_CD_LPIPS_multi_step.sh

## Random pixel inpainting (10%)
CPS_Diverse/scripts/fidelity_exps/solve_pixel_inpaint_our_method_CD_LPIPS_10_one_step.sh
CPS_Diverse/scripts/fidelity_exps/solve_pixel_inpaint_our_method_CD_LPIPS_10_multi_step.sh


#------------------------------------ ImageNet (64 x 64) ------------------------------------ 

## 4x Super-Resolution
CPS_Diverse/scripts/fidelity_exps/solve_SR_our_method_CD_LPIPS_4x_one_step.sh
CPS_Diverse/scripts/fidelity_exps/solve_SR_our_method_CD_LPIPS_4x_multi_step.sh

## Gaussian Deblur
CPS_Diverse/scripts/fidelity_exps/solve_ImNet_GD_our_method_CD_LPIPS_one_step.sh
CPS_Diverse/scripts/fidelity_exps/solve_ImNet_GD_our_method_CD_LPIPS_multi_step.sh

## Random pixel inpainting (20%)
CPS_Diverse/scripts/fidelity_exps/solve_pixel_inpaint_our_method_CD_LPIPS_20_one_step.sh
CPS_Diverse/scripts/fidelity_exps/solve_pixel_inpaint_our_method_CD_LPIPS_20_multi_step.sh

```
## ⚙️ Diversity Experiments
```bash
# LSUN-Beadroom (256 x 256)

## 8x Super-Resolution
CPS_Diverse/scripts/diversity_exps/solve_SR_our_method_CD_LPIPS_8x_one_step.sh
CPS_Diverse/scripts/diversity_exps/solve_SR_our_method_CD_LPIPS_8x_multi_step.sh

## Gaussian Deblur
CPS_Diverse/scripts/diversity_exps/solve_GD_our_method_CD_LPIPS_one_step.sh
CPS_Diverse/scripts/diversity_exps/solve_GD_our_method_CD_LPIPS_multi_step.sh

## Random pixel inpainting (10%)
CPS_Diverse/scripts/diversity_exps/solve_pixel_inpaint_our_method_CD_LPIPS_10_one_step.sh
CPS_Diverse/scripts/diversity_exps/solve_pixel_inpaint_our_method_CD_LPIPS_10_multi_step.sh


```

## Acknowledgments

Thanks to the open source codebases. Our codebase is built on them.