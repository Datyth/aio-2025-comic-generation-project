from pydantic import BaseModel
from typing import List, Optional

class ComicRequest(BaseModel):
    story_text: str
    max_scenes: Optional[int] = 4

class SceneResponse(BaseModel):
    scene_index: int
    prompt: str
    image_base64: str

class ComicResponse(BaseModel):
    scenes: List[SceneResponse]
