import torch
import numpy as np
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from utils import *
import argparse, os, json
device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""
Sample command line prompt 
python run_noising_denoising.py --prompt "a highly realistic photo of a car" --seed 224 --total_time 100 --model_type stableV2.1 --orig_directory /GENERATED/IMAGES/HERE --output_directory  /GENERATED/NOISED_IMAGES/HERE --to_time_step_t 1 --new_images_per_old 1
"""

def main():
    torch.set_grad_enabled(False)
    parser = argparse.ArgumentParser()  
    parser.add_argument('--prompt', action="store", type=str, required=True, help='Prompt to use')
    parser.add_argument('--seed', action="store", type=int, required=True,help='Seed')
    parser.add_argument('--total_time', action="store", type=int, required=True,help='total time of SDE')
    parser.add_argument('--model_type', action="store", type=str, required=True, help="model_name")
    parser.add_argument('--guidance_scale', action="store", type=int, default=7.5,required=False, help="guidance scale")

    parser.add_argument('--orig_directory', action="store", type=str, required=True, help="location of original images/latents")
    parser.add_argument('--output_directory', action="store", type=str, required=True, help="location of new images")
    parser.add_argument('--to_time_step_t', action="store",type=int, required=True, help="Number of steps to noise")
    parser.add_argument('--new_images_per_old', action="store",type=int, required=True, help="Number of new images per old image")
    parser.add_argument('--image_num', action="store",type=int, required=False, default=None, help="Image number")
    args = parser.parse_args()

    prompt = args.prompt
    seed = args.seed
    total_time = args.total_time
    model_type = args.model_type
    guidance_scale = args.guidance_scale
    
    orig_directory = args.orig_directory
    output_directory = args.output_directory
    to_time_step_t = args.to_time_step_t
    new_images_per_stuff = args.new_images_per_old
    image_num = args.image_num

    print(f"RUNNING run_noising_denoising.py ON PROMPT {prompt}, DATA {orig_directory}, DESTINATION {output_directory}, TOTAL_TIME {total_time}, TIME_NOISE {to_time_step_t}, NEW_IMAGES {new_images_per_stuff}, IMAGE_NUM {image_num}")
    print("LOADING MODELS")
    
    vae, tokenizer, text_encoder, unet, scheduler = load_model(model_type,total_time,device)

    print("FINISHED LOADING MODELS")

    print("GENERATING NOISY_DENOISY WITH PATHS")
    cl = NoiseDenoise(vae, tokenizer, text_encoder, unet, scheduler, device,num_inference_steps=total_time,seed=seed,guidance_scale=guidance_scale)
    cl.generate_noised_denoise_images(prompt, orig_directory, output_directory, to_time_step_t, new_images_per_stuff,image_num)

if __name__ == "__main__":
    main()

