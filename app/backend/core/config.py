import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

env_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env")
load_dotenv(env_path)

class Settings(BaseSettings):
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")
    CACHE_DIR: str = os.getenv("CACHE_DIR", "./cache")
    MAX_SCENES: int = 4
    COMIC_STYLE: str = (
        "whimsical watercolor children's book illustration, "
        "hand-drawn, soft pencil outlines, pastel colors, "
        "cozy atmosphere, high quality, magical lighting"
    )

settings = Settings()