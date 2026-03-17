from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging
import os

from api.routers import router
from core.config import settings

from models.llm import LLMClient
from models.stable_diffusion import DiffusionClient
from modules.story_generator import StoryGenerator
from modules.image_prompt_generator import ImagePromptGenerator
from modules.image_generator import ImageGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Determine absolute path for cache_dir if it's relative
    cache_dir = settings.CACHE_DIR
        
    logger.info("Downloading/Loading LLM Model ...")
    app.state.llm_client = LLMClient(hf_token=settings.HF_TOKEN, cache_dir=cache_dir)
    app.state.llm_client.load_model()
    
    app.state.story_generator = StoryGenerator(llm_client=app.state.llm_client)
    app.state.image_prompt_gen = ImagePromptGenerator(app.state.llm_client)
    
    logger.info("Downloading/Loading Stable Diffusion Model ...")
    app.state.diffusion_client = DiffusionClient(hf_token=settings.HF_TOKEN, cache_dir=cache_dir)
    app.state.diffusion_client.load_model()
    app.state.image_generator = ImageGenerator(diffusion_client=app.state.diffusion_client)
    
    logger.info("Downloading/Loading models completed!")
    
    yield 
    
    logger.info("Cleaning memory...")
    app.state.llm_client = None
    app.state.diffusion_client = None

app = FastAPI(
    title="Comic Generation API", 
    description="Automatical comic generation by LLM and Diffusion",
    lifespan=lifespan
)

app.include_router(router, prefix="/api/v1")