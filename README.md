# Critical windows

By Marvin Li and Sitan Chen

## Overview
This repo provides some experimental tools to investigate the phenomena of feature emergence in diffusion models, where features of the final outputted images like color, background, or clothing type are fossilized in narrow intervals of the reverse denoising process. This code accompanies the **ICML 2024** paper [(Li and Chen, 2024)](https://arxiv.org/pdf/2403.01633).

## Installation

To download from our github, use
```bash
git clone https://github.com/marvinli-harvard/critical-windows
cd critical-windows
pip install -r requirements.txt
```

## Quickstart
There are two main experiment scripts that we use to produce the results. We first need the script ``run_data_collection.py`` that generates images from a diffusion model for a given prompt. 
```bash
python run_data_collection.py --prompt "a highly realistic photo of a car" --location /GENERATED/IMAGES/HERE --seed 224 --total_time 100 --nimages 100 --model_type stableV2.1
```
Then we generate new images from the above directory of images by noising and denoising them, look at ``run_noising_denoising.py``.
```bash
python run_noising_denoising.py --prompt "a highly realistic photo of a car" --seed 224 --total_time 100 --model_type stableV2.1 --orig_directory /GENERATED/IMAGES/HERE --output_directory  /NOISED/IMAGES/HERE/ --to_time_step_t 10 --new_images_per_old 1
```
If you want to instead take the **trajectory** of a given image and generate new images from that, we can use the following command and the file ``from_gaussian_noise_and_clone.py``.
```bash
python run_clone_denoising.py --prompt "a highly realistic photo of a car" --destination /CLONED/IMAGES/HERE --seed 210 --total_time 100 --nimages 100 --clone_at_time 10 --model_type stableV2.1
```

## Citation
If you use our code or otherwise find this library useful, please cite:
```
@misc{li2024criticalwindowsnonasymptotictheory,
      title={Critical windows: non-asymptotic theory for feature emergence in diffusion models}, 
      author={Marvin Li and Sitan Chen},
      year={2024},
      eprint={2403.01633},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2403.01633}, 
}
```
