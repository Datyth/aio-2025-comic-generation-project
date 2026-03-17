import os
import sys
import json

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
hf_token = os.getenv("HF_TOKEN", "")
cache_dir = os.getenv("CACHE_DIR", "../../cache")

from llm import LLMClient
from stable_diffusion import DiffusionClient
from story_generator import StoryGenerator
from image_prompt_generator import ImagePromptGenerator
from image_generator import ImageGenerator

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(RESULTS_DIR, exist_ok=True)

story_text = "Một con quạ khát nước tìm thấy một bình nước nhưng cổ bình quá cao. Quạ thông minh đã gắp từng hòn sỏi bỏ vào bình để nước dâng lên và uống được."

def save_json(filename: str, data: dict):
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved results to: {path}")

def run_pipeline():
    comic_style = "whimsical fable book illustration, highly detailed, vibrant colors, fantasy art style"

    print("\n[1/3] Generating story scenes...")
    llm_client = LLMClient(hf_token=hf_token, cache_dir=cache_dir)
    llm_client.load_model()
    
    story_generator = StoryGenerator(llm_client=llm_client)
    
    generated_story_text = story_generator.gen_story_structured(story_text, max_new_tokens=400)
    print(f"\nGenerated Texts:\n{generated_story_text}\n")

    print("\n[2/3] Generating image prompts...")
    image_prompt_gen = ImagePromptGenerator(llm_client)
    paragraphs = image_prompt_gen.parse_paragraphs(generated_story_text)
    
    if not paragraphs:
        print("Error format <|paragraph|>... using fallback.")
        paragraphs = [p.strip() for p in generated_story_text.split('.') if len(p.strip()) > 10][:2]
        
    print("\n[3/3] Generating images with Stable Diffusion XL...")
    diffusion_client = DiffusionClient(hf_token=hf_token, cache_dir=cache_dir)
    diffusion_client.load_model()
    image_generator = ImageGenerator(diffusion_client=diffusion_client)

    saved_metadata = []
    for i, para in enumerate(paragraphs):
        visual_prompt = image_prompt_gen.create_visual_prompt(
            paragraph=para,
            full_story=story_text,
            style=comic_style
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
            filename = f"scene_{i+1:02d}.png"
            filepath = os.path.join(RESULTS_DIR, filename)
            img.save(filepath)
            print(f"  → Saved: {filepath}")
            saved_metadata.append({
                "scene_index": i,
                "paragraph": para,
                "filename": filename,
                "prompt": visual_prompt
            })

    save_json("pipeline_result.json", {"generated_images": saved_metadata})
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    print("=" * 40)
    print("Running Full Comic Generation Pipeline in comic_generation_layout")
    print("=" * 40)
    run_pipeline()