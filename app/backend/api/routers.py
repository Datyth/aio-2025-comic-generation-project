from fastapi import APIRouter, Request, HTTPException
from schemas.payload import ComicRequest, ComicResponse, SceneResponse
from core.config import settings
import base64
from io import BytesIO
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

def pil_image_to_base64(image) -> str:
    buffered = BytesIO()
    image.save(buffered, format = "PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

@router.post("/generate-comic", response_model=ComicResponse)
async def generate_comic(request: Request, payload: ComicRequest):
    try:
        # Get downloaded models
        story_generator = request.app.state.story_generator
        image_prompt_gen = request.app.state.image_prompt_gen
        image_generator = request.app.state.image_generator
        
        # 1. Generate story text
        print("[1/3] Generating story structure...")
        story_text = story_generator.gen_story_structured(
            payload.story_text, 
            max_new_tokens=400
        )
        
        paragraphs = image_prompt_gen.parse_paragraphs(story_text)
        if not paragraphs:
            print("Error format <|paragraph|>... using fallback.")
            paragraphs = [p.strip() for p in story_text.split('.') if len(p.strip()) > 10][:2]
        
        print(f"[2/3] Generating image prompts for {len(paragraphs)} scenes...")
        
        response_scenes = []
        for i, para in enumerate(paragraphs):
            visual_prompt = image_prompt_gen.create_visual_prompt(
                paragraph=para,
                full_story=payload.story_text,
                style=settings.COMIC_STYLE
            )
            
            print(f"--- Scene {i+1} ---")
            print(f"Caption: {para}")
            print(f"Image Prompt: {visual_prompt}\n")

            img = image_generator.generate_image(
                prompt=visual_prompt,
                paragraph=para,
                num_inference_steps=5,
                guidance_scale=2.0,
                size=1024
            )
            
            if img:
                img_b64 = pil_image_to_base64(img)
                response_scenes.append(
                    SceneResponse(
                        scene_index = i,
                        paragraph = para,
                        prompt = visual_prompt,
                        image_base64 = img_b64
                    )
                )
            
        return ComicResponse(scenes = response_scenes)
        
    except Exception as e:
        print(f"Error during generation: {e}")
        raise HTTPException(status_code = 500, detail=f"Error: {str(e)}")