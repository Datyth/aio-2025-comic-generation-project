from fastapi import APIRouter, Request, HTTPException
from schemas.payload import ComicRequest, ComicResponse, SceneResponse
from core.config import settings
import base64
from io import BytesIO

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
        
        #Generate
        print("[1/3] Generating story scenes...")
        scenes, characters = story_generator.gen_story(
            payload.story_text, 
            max_scenes=payload.max_scenes
        )
        
        print("[2/3] Generating image prompts...")
        image_prompts = image_prompt_gen.generate_prompts(
            scenes, 
            characters, 
            style=settings.COMIC_STYLE
        )
        
        print("[3/3] Generating images...")
        generation_results = image_generator.generate_images(
            image_prompts = image_prompts,
            num_inference_steps = 10,
            guidance_scale = 7.5,
            size = 1024
        )
        
        #Sent to frontend.
        response_scenes = []
        for res in generation_results:
            img_b64 = pil_image_to_base64(res["image"])
            response_scenes.append(
                SceneResponse(
                    scene_index = res["scene_index"],
                    prompt = res["prompt"],
                    image_base64 = img_b64
                )
            )
            
        return ComicResponse(scenes = response_scenes)
        
    except Exception as e:
        print(f"Error during generation: {e}")
        raise HTTPException(status_code = 500, detail=f"Error: {str(e)}")