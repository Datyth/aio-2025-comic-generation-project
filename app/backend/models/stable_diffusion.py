import torch
import logging
from typing import Optional
from diffusers import StableDiffusionXLPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DiffusionClient:
    def __init__(
        self, 
        huggingface_path: str = "Lykon/dreamshaper-xl-v2-turbo", 
        hf_token: str = None,
        cache_dir: Optional[str] = "./cache"
    ):
        self.huggingface_path = huggingface_path
        self.hf_token = hf_token
        self.cache_dir = cache_dir
        self.pipeline = None

    def load_model(self):
        logger.info("Loading SDXL model...")
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            self.huggingface_path,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            token=self.hf_token,
            cache_dir=self.cache_dir
        )
        self.pipeline.enable_model_cpu_offload()
        self.pipeline.enable_vae_slicing()
        self.pipeline.enable_vae_tiling()
        logger.info(f"Loaded SDXL model {self.huggingface_path} successfully.")

    def gen_image(self, prompt: str, negative_prompt: str = "", num_inference_steps: int = 5, guidance_scale: float = 2.0, width: int = 1024, height: int = 1024):
        if self.pipeline is None:
            self.load_model()
            
        image = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height
        ).images[0]
        return image