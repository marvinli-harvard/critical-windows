import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel, DDPMScheduler, DDIMScheduler
from pathlib import Path
from PIL import Image
import os
from typing import Tuple, List, Optional

####################################################################################################
# BASIC UTILITIES
####################################################################################################
def load_pt_file(file_path : str ,t : int = 0):
    tensor = torch.load(file_path)
    return tensor[-(t+1),:, :, :].numpy()  # Extract the last slice of the first index and convert to numpy

def create_directory(path : str):
    if not os.path.exists(path):
        os.makedirs(path)

def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """
    Convert a PIL Image to a PyTorch Tensor without using torchvision.

    Args:
        image (Image.Image): A PIL Image object to be converted.

    Returns:
        Tensor: A PyTorch tensor representing the image in (C x H x W) format 
                with pixel values normalized to the range [0, 1].
    """
    # Convert the PIL image to a NumPy array
    numpy_image = np.array(image)

    # Convert the NumPy array to a PyTorch tensor
    tensor_image = torch.tensor(numpy_image)

    # Convert the tensor to float and scale it to [0, 1]
    tensor_image = tensor_image.float() / 255.0

    # Change the layout from (H x W x C) to (C x H x W)
    if tensor_image.ndimension() == 3:
        tensor_image = tensor_image.permute(2, 0, 1)

    return tensor_image


def load_model(model_type : str,
               total_time : int,
               device : str,
               ddim : bool = False
    )->Tuple[AutoencoderKL, CLIPTokenizer, CLIPTextModel, UNet2DConditionModel, DDPMScheduler]:
    """
    Load diffusion model from name 

    Args:
        model_type (str): String of model to download
        total_time (int): Number of steps of denoising
        device (str): Load models on here
        ddim (bool): DDIM sampling

    Returns:
        Tuple[AutoencoderKL, CLIPTokenizer, CLIPTextModel, UNet2DConditionModel, DDPMScheduler]
    """
    if model_type == "stableV2.1":
        link = "stabilityai/stable-diffusion-2-1"
        # 1. Load the autoencoder model which will be used to decode the latents into image space.
        vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1",subfolder="vae").to(device)
    elif model_type == "stableXL":
        assert False 
        link = "stabilityai/stable-diffusion-xl-base-1.0"
    elif model_type=="animagine":
        link = "cagliostrolab/animagine-xl-3.0"
        # 1. Load the autoencoder model which will be used to decode the latents into image space.
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix").to(device)

    # 2. Load the tokenizer and text encoder to tokenize and encode the text.
    tokenizer = CLIPTokenizer.from_pretrained(link,subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(link,subfolder="text_encoder").to(device)

    # 3. The UNet model for generating the latents.
    unet = UNet2DConditionModel.from_pretrained(link, subfolder="unet").to(device)

    # 4. The Scheduler for generating the images.
    if ddim:
        print("USING DDIM")
        ddim_scheduler_config = {
                "_class_name": "DDIMScheduler",
                "_diffusers_version": "0.8.0",
                "beta_end": 0.012,
                "beta_schedule": "scaled_linear",
                "beta_start": 0.00085,
                "clip_sample": False,
                "num_train_timesteps": 1000,
                "prediction_type": "v_prediction",
                "set_alpha_to_one": False,
                "skip_prk_steps": True,
                "steps_offset": 1,
                "trained_betas": None
                }
        scheduler = DDIMScheduler.from_config(ddim_scheduler_config, rescale_betas_zero_snr=True, timestep_spacing="trailing")        
    else:
        scheduler = DDPMScheduler.from_pretrained(link, subfolder="scheduler")
    scheduler.set_timesteps(total_time)
    
    return vae, tokenizer, text_encoder, unet, scheduler

####################################################################################################
# EXPERIMENT CLASSES
####################################################################################################

class DiffusionPipeline:
    """
    Base class for all sampling experiments. 

    Attributes:
        vae (AutoencoderKL): image encoder and decoder
        tokenizer (ClipTokenizer): prompt tokenizer
        text_encoder (ClipTextModel): token encoder
        unet (UNet2DConditionalModel): unet to encode text
        scheduler (DDPMScheduler): diffusion model scheduler
        device (str) : where to place models
        num_inference_steps (int): number of steps of diffusion
        guidance_scale (float) : guidance scale
        seed (int): for reproducibility
        pipeline (StableDiffusionPipeline) : diffusion model pipeline
    """

    def __init__(self, 
                 vae : AutoencoderKL, 
                 tokenizer : CLIPTokenizer, 
                 text_encoder : CLIPTextModel, 
                 unet : UNet2DConditionModel, 
                 scheduler : DDPMScheduler,  
                 device : str, 
                 num_inference_steps : int ,
                 guidance_scale : float,
                 seed : int=224):
        """
        Initialize DiffusionPipeline object. 

        Args:
            vae (AutoencoderKL): image encoder and decoder
            tokenizer (ClipTokenizer): prompt tokenizer
            text_encoder (ClipTextModel): token encoder
            unet (UNet2DConditionalModel): unet to encode text
            scheduler (DDPMScheduler): diffusion model scheduler
            device (str) : where to place models
            num_inference_steps (int): number of steps of diffusion
            guidance_scale (float) : guidance scale
            seed (int): for reproducibility
        """
        self.vae = vae
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.unet = unet
        self.scheduler = scheduler
        self.num_inference_steps = num_inference_steps
        if self.scheduler:
            self.scheduler.set_timesteps(num_inference_steps)
        self.device = device

        self.pipeline = StableDiffusionPipeline(self.vae,self.text_encoder,self.tokenizer,self.unet,self.scheduler,None,None)
        
        torch.manual_seed(seed)
        self.generator = torch.Generator(device="cuda").manual_seed(seed)
        self.guidance_scale = guidance_scale
            
    ## Taken from https://wandb.ai/capecape/ddpm_clouds/reports/Using-Stable-Diffusion-VAE-to-encode-satellite-images--VmlldzozNDA2OTgx
    def encode(self,input_images:torch.Tensor)->torch.Tensor:
        """
        Encode input images with vae

        Args:
            input_images (torch.Tensor): Images to encode 

        Returns:
            torch.Tensor : vae output
        """
        if input_images.shape[1]==1:
            input_images = input_images.repeat(1,3,1,1)
        with torch.no_grad():
            latent = self.vae.encode(input_images*2 - 1) # Note scaling
        return self.vae.config.scaling_factor * latent.latent_dist.sample()

    def decode(self, latents : torch.Tensor)->Image:
        """
        Decode images with vae

        Args:
            latents (torch.Tensor): Latents to decode

        Returns:
            Image : outputted images
        """
        latents = (1 / self.vae.config.scaling_factor) * latents
        with torch.no_grad():
            image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1).squeeze()
        image = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
        image = Image.fromarray(image)
        return image

class GenerateImagesWithPaths(DiffusionPipeline):
    """
    Generate images and paths
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args, **kwargs)        

    def generate_images_with_paths(self,
                                   prompt :str, 
                                   nexamples :int,
                                   location:str):
        """
        Generate image from `prompt` and place the path of sample in `location`
        Args:
            prompt (str) : prompt to use
            nexamples (int) : Number of images to consider
            location (str) : output directory
        Return Tuple[List[Image], List[torch.Tensor]]
        """
        images = []
        embeddings = []
        for i in range(nexamples):
            with torch.no_grad():
                text_input = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
                max_length = text_input.input_ids.shape[-1]
                uncond_input = self.tokenizer([""], padding="max_length", max_length=max_length, return_tensors="pt")
                
                text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
                uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
                text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

                latents = torch.randn(
                        (1, self.unet.config.in_channels, 512 // 8, 512 // 8),
                        generator=self.generator,
                        device=self.device,
                        )
                latents = latents * self.scheduler.init_noise_sigma
                history_latent = [latents.clone().cpu()]
                for t in tqdm(self.scheduler.timesteps):
                    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                    latent_model_input = torch.cat([latents] * 2)
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)

                    # predict the noise residual
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings,added_cond_kwargs={}).sample

                    # perform guidance
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents,generator=self.generator).prev_sample
                    history_latent.append(latents.clone().cpu())
                
                history_latent = torch.concat(history_latent).cpu()
                img = self.decode(latents)
                img.save(location+f"{i}.PNG")
                torch.save(history_latent, location+f"{i}.pt")

                embeddings.append(history_latent)
                images.append(img)
        return images, embeddings

class NoiseDenoise(DiffusionPipeline):
    """
    Noise and then denoise experiment
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args, **kwargs)        
    
    def generate_noised_denoise_images(self, 
                                       prompt :str, 
                                       orig_directory:str, 
                                       output_directory:str, 
                                       to_time_step_t:int, 
                                       new_images_per_stuff:int, 
                                       image_num:Optional[int]=None
        ) ->None:
        """
        Noise images from `orig_directory` to a given timestep and then denoise
        and place into `output_directory`
        Args:
            prompt (str) : prompt to use
            orig_directory (str) : original directory
            output_directory (str) : output directory
            to_time_step_t (int) : timestep to which to noise
            new_images_per_stuff (int) : Number of new images per original image
            image_num (Optional[int]) : specific image to consider
        """
        if image_num is None:
            ## Make output directory and files
            files = [f for f in os.listdir(orig_directory) if f.endswith('.pt') and int(f.split(".pt")[0]) in list(range(10))]
            files.sort()
        else:
            files = [f"{image_num}.pt"]
        
        for f in files:
            create_directory(os.path.join(output_directory,f"time_{to_time_step_t}",Path(f).stem))
        
        ## Load embedding for each f in files and run noise/denoise process up to timestep_t
        text_input = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer([""], padding="max_length", max_length=max_length, return_tensors="pt")
        
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        for f in files:
            image_embedding = torch.tensor(load_pt_file(os.path.join(orig_directory,f))).to(self.device)[None,:,:,:]
            for i in range(new_images_per_stuff):                
                noise = torch.randn(image_embedding.shape)
                latents = self.scheduler.add_noise(image_embedding, noise.to(self.device),  timesteps=torch.tensor([self.scheduler.timesteps[-to_time_step_t]]).to(self.device))
                history_latent = [latents.clone().cpu()]
                for _, t in tqdm(enumerate(self.scheduler.timesteps[-to_time_step_t:]),total=to_time_step_t):
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2)
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)

                    # predict the noise residual
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

                    # perform guidance
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents,generator=self.generator).prev_sample
                    history_latent.append(latents.clone().cpu())

                history_latent = torch.concat(history_latent).cpu()
                img = self.decode(latents)
                img.save(os.path.join(output_directory,f"time_{to_time_step_t}",Path(f).stem,f"{i}.png"))
                torch.save(history_latent, os.path.join(output_directory,f"time_{to_time_step_t}",Path(f).stem,f"{i}.pt"))
              

class NoiseClone(DiffusionPipeline):
    """
    Noise and then clone experiment
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args, **kwargs)        

    def generate_images_and_clones(self, 
                                   prompt : str,
                                   nexamples : int,
                                   location : str,
                                   clone_at_time : int
        )->None:
        """
        Generate image from `prompt` and clone it at time `clone_at_time`
        Args:
            prompt (str) : prompt to use
            nexamples (int) : Number of images to consider
            location (str) : output directory
            clone_at_time (int) : timestep at which to clone
        """
        for i in range(nexamples):
            with torch.no_grad():
                text_input = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
                max_length = text_input.input_ids.shape[-1]
                uncond_input = self.tokenizer([""], padding="max_length", max_length=max_length, return_tensors="pt")
                
                text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
                uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
                text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

                latents = torch.randn(
                        (1, self.unet.config.in_channels, 512 // 8, 512 // 8),
                        generator=self.generator,
                        device=self.device,
                        )
                latents = latents * self.scheduler.init_noise_sigma
                history_latent = [latents.clone().cpu()]
                old_latents = None 
                
                for t_ind,t in tqdm(list(enumerate(self.scheduler.timesteps))):
                    if t_ind==clone_at_time:   
                        print(t_ind,t)
                        old_latents = latents.clone().cpu()

                    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                    latent_model_input = torch.cat([latents] * 2)
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)

                    # predict the noise residual
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings,added_cond_kwargs={}).sample

                    # perform guidance
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents,generator=self.generator).prev_sample
                    history_latent.append(latents.clone().cpu())
                
                history_latent = torch.concat(history_latent).cpu()
                img = self.decode(latents)
                img.save(location+f"{i}_cloneat={clone_at_time}_orig.PNG")
                torch.save(history_latent, location+f"{i}_cloneat={clone_at_time}_orig.pt")

                
                history_latent = [latents.clone().cpu()]
                latents = old_latents.clone().cuda()
                for t in tqdm(self.scheduler.timesteps[clone_at_time:]):
                    
                    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                    latent_model_input = torch.cat([latents] * 2)
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)

                    # predict the noise residual
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings,added_cond_kwargs={}).sample

                    # perform guidance
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents,generator=self.generator).prev_sample
                    history_latent.append(latents.clone().cpu())
                
                history_latent = torch.concat(history_latent).cpu()
                img = self.decode(latents)
                img.save(location+f"{i}_cloneat={clone_at_time}_clone.PNG")
                torch.save(history_latent, location+f"{i}_cloneat={clone_at_time}_clone.pt")

                
        return 
    

