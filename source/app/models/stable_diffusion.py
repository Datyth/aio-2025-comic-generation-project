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

        self.model_path = huggingface_path
        self.hf_token = hf_token
        self.cache_dir = cache_dir
        self.sd_pipeline = self.prepare_model()
        logger.info(f"Loaded SDXL model {huggingface_path} successfully.")

    def prepare_model(self):
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            token=self.hf_token,
            cache_dir=self.cache_dir
        ).to("cuda")

        pipeline.enable_model_cpu_offload()
        return pipeline

    def gen_image(self, prompt: str, negative_prompt: str, num_inference_steps: int, guidance_scale: float, width: int = 1024, height: int = 1024):
        image = self.sd_pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height
        ).images[0]
        return image