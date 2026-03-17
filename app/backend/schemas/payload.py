from pydantic import BaseModel
from typing import List, Optional

class ComicRequest(BaseModel):
    story_text: str

class SceneResponse(BaseModel):
    scene_index: int
    paragraph: str
    prompt: str
    image_base64: str

class ComicResponse(BaseModel):
    scenes: List[SceneResponse]
