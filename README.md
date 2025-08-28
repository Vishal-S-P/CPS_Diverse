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
| LSUN-Bedroom     |  |     |
| ImageNet   |    |     |

Download datasets from following links and place them in 'sample_dataset' folder -

| Dataset | Link |
| :------- |:--------:| 
| LSUN-Bedroom     |  
| ImageNet   |    |     

## Fidelity Experiments
```bash
# 8x Super-Resolution

### Linear inverse problem

### Non-Linear inverse problem


## Diversity Experiments


### Linear inverse problem

### Non-Linear inverse problem

## Acknowledgments

Thanks to the open source codebases. Our codebase is built on them.