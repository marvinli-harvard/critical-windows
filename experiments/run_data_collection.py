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
python run_data_collection.py --prompt "a highly realistic photo of a car" --location cars/ --seed 224 --total_time 10 --nimages 1 --model_type stableV2.1
"""

def main():
    torch.set_grad_enabled(False)
    parser = argparse.ArgumentParser()  
    parser.add_argument('--prompt', action="store", type=str, required=True, help='Prompt to use')
    parser.add_argument('--location', action="store", type=str, required=True,help='Location of image/embedding pairs')
    parser.add_argument('--seed', action="store", type=int, required=True,help='Seed')
    parser.add_argument('--total_time', action="store", type=int, required=True,help='total time of SDE')
    parser.add_argument('--nimages', action="store", type=int, required=True,help='number of images')
    parser.add_argument('--model_type', action="store", type=str, required=True, help="model_name")
    parser.add_argument('--guidance_scale', action="store", type=int, default=7.5,required=False, help="guidance scale")
    parser.add_argument('--ddim', action="store_true", required=False, help="ddim scheduler")
    args = parser.parse_args()

    prompt = args.prompt
    location = args.location
    seed = args.seed
    total_time = args.total_time
    nimages = args.nimages
    model_type = args.model_type
    guidance_scale = args.guidance_scale
    ddim = args.ddim

    print(f"RUNNING ON PROMPT {prompt}, DESTINATION {location}, TOTAL_TIME {total_time}, NIMAGES = {nimages}, DDIM={ddim}")
    print("LOADING MODELS")
    
    vae, tokenizer, text_encoder, unet, scheduler = load_model(model_type,total_time,device, ddim)

    print("FINISHED LOADING MODELS")

    print("GENERATING IMAGES WITH PATHS")    
    create_directory(location)
    cl = GenerateImagesWithPaths(vae, tokenizer, text_encoder, unet, scheduler, device,num_inference_steps=total_time,seed=seed,guidance_scale=guidance_scale)
    cl.generate_images_with_paths(prompt, nimages, location)

if __name__ == "__main__":
    main()

