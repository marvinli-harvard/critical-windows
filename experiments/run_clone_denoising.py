import torch
import numpy as np
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from utils import *
import argparse, os
device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""
Sample command line prompt 
python run_clone_denoising.py --prompt "a highly realistic photo of a car" --destination cars_clone/ --seed 210 --total_time 500 --nimages 1 --clone_at_time 5 --model_type stableV2.1
"""

def main():    
    torch.set_grad_enabled(False)
    parser = argparse.ArgumentParser()  
    parser.add_argument('--prompt', action="store", type=str, required=True, help='Prompt to use')
    parser.add_argument('--destination', action="store", type=str, required=True, help='Destination of new images')
    parser.add_argument('--seed', action="store", type=int, required=True,help='Seed (for reproducibility)')
    parser.add_argument('--total_time', action="store", type=int, required=True,help='Total number of steps of denoising')
    parser.add_argument('--nimages', action="store", type=int, required=True,help='Number of images to generate')
    parser.add_argument('--clone_at_time', action="store", type=int, required=True,help='Which time to clone image at')
    parser.add_argument('--model_type', action="store", type=str, required=True, help="Name of model")
    parser.add_argument('--guidance_scale', action="store", type=float, default=7.5,required=False, help="Guidance scale")
    
    args = parser.parse_args()
    prompt = args.prompt
    destination = args.destination
    seed = args.seed
    total_time = args.total_time
    nimages = args.nimages
    clone_at_time = args.clone_at_time
    model_type = args.model_type
    guidance_scale = args.guidance_scale

    print(f"RUNNING ON PROMPT {prompt}, DESTINATION {destination}, TOTAL_TIME {total_time}, NIMAGES = {nimages}, CLONE_TIME {clone_at_time}")
    print("LOADING MODELS")
    
    vae, tokenizer, text_encoder, unet, scheduler = load_model(model_type,total_time,device)

    print("FINISHED LOADING MODELS")

    print("GENERATING IMAGES WITH PATHS")
    
    create_directory(destination)
    cl = NoiseClone(vae, tokenizer, text_encoder, unet, scheduler, device,num_inference_steps=total_time,seed=seed,guidance_scale=guidance_scale)
    cl.generate_images_and_clones(prompt, nimages, destination, clone_at_time)

if __name__ == "__main__":
    main()

