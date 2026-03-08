import os
import json
from dotenv import load_dotenv
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from app.models.llm import LLMClient
from app.models.stable_diffusion import DiffusionClient
from app.modules.story_generator import StoryGenerator
from app.modules.image_prompt_generator import ImagePromptGenerator
from app.modules.image_generator import ImageGenerator

load_dotenv()
hf_token = os.getenv("HF_TOKEN", "")
cache_dir = os.getenv("CACHE_DIR", "./cache")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

story_text = """
Cậu bé vốn hay bỏ dở việc học, làm gì cũng chóng chán. Một hôm, cậu thấy một bà cụ kiên trì mài thỏi sắt thành kim, cậu ngạc nhiên và được bà nói rằng kiên trì mỗi ngày sẽ đạt kết quả. Bài học khiến cậu bé nhận ra giá trị của sự bền bỉ, từ đó chăm chỉ học tập và không còn bỏ dở như trước.
"""

def save_json(filename: str, data: dict):
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved results to: {path}")

def run_story_gen():
    llm_client = LLMClient(hf_token=hf_token, cache_dir=cache_dir)
    story_generator = StoryGenerator(llm=llm_client)

    scenes, characters = story_generator.gen_story(story_text, max_scenes=4)

    print("Story Scenes:")
    for i, scene in enumerate(scenes):
        print(f"Scene {i+1}:")
        print(f"  Description: {scene.description}")
        print(f"  Characters: {', '.join(scene.characters)}")
        print(f"  Action: {scene.action}")
        print(f"  Setting: {scene.setting}")
        print()

    print("Characters:")
    for character in characters:
        print(f"Name: {character.name}")
        print(f"  Appearance: {character.appearance}")
        print(f"  Personality: {character.personality}")
        print(f"  Role: {character.role}")
        print()

    # Save to JSON
    result = {
        "scenes": [
            {
                "description": s.description,
                "characters": s.characters,
                "action": s.action,
                "setting": s.setting,
            }
            for s in scenes
        ],
        "characters": [
            {
                "name": c.name,
                "appearance": c.appearance,
                "personality": c.personality,
                "role": c.role,
            }
            for c in characters
        ],
    }
    save_json("story_gen_result.json", result)

    return scenes, characters


def run_image_prompt():
    llm_client = LLMClient(hf_token=hf_token, cache_dir=cache_dir)
    story_generator = StoryGenerator(llm=llm_client)
    scenes, characters = story_generator.gen_story(story_text, max_scenes=4)

    try:
        image_prompt_gen = ImagePromptGenerator(llm_client)

        comic_style = "whimsical watercolor children's book illustration, hand-drawn, soft pencil outlines, pastel colors, cozy atmosphere, high quality, magical lighting"
        image_prompts = image_prompt_gen.generate_prompts(scenes, characters, style=comic_style)

        result = []
        for prompt_obj in image_prompts:
            print(f"--- Scene {prompt_obj.scene_index + 1} ---")
            print(f"[Character Context]: {prompt_obj.char_contexts}")
            print(f"[Final Prompt]: {prompt_obj.prompt}")
            print(f"[Negative Prompt]: {prompt_obj.negative_prompt}")
            print("-" * 30 + "\n")

            result.append({
                "scene_index": prompt_obj.scene_index,
                "description": prompt_obj.description,
                "character_context": prompt_obj.char_contexts,
                "prompt": prompt_obj.prompt,
                "negative_prompt": prompt_obj.negative_prompt,
            })

        # Save to JSON
        save_json("image_prompt_result.json", {"image_prompts": result})

    except Exception as e:
        print(f"Error: {e}")

def run_pipeline():
    """
    Full pipeline:
      1. Generate story scenes & characters from raw text (LLM)
      2. Generate image prompts for each scene (LLM)
      3. Generate images with captions (Stable Diffusion XL)
      4. Save every image as PNG to RESULTS_DIR
    """
    comic_style = (
        "whimsical watercolor children's book illustration, "
        "hand-drawn, soft pencil outlines, pastel colors, "
        "cozy atmosphere, high quality, magical lighting"
    )

    print("\n[1/3] Generating story scenes and characters...")
    llm_client = LLMClient(hf_token=hf_token, cache_dir=cache_dir)
    story_generator = StoryGenerator(llm=llm_client)
    scenes, characters = story_generator.gen_story(story_text, max_scenes=4)

    print(f"  → {len(scenes)} scenes, {len(characters)} characters extracted.")
    for i, scene in enumerate(scenes):
        print(f"     Scene {i+1}: {scene.description[:60]}...")

    print("\n[2/3] Generating image prompts...")
    image_prompt_gen = ImagePromptGenerator(llm_client)
    image_prompts = image_prompt_gen.generate_prompts(scenes, characters, style=comic_style)
    print(f"  → {len(image_prompts)} image prompts generated.")

    print("\n[3/3] Generating images with Stable Diffusion XL...")
    diffusion_client = DiffusionClient(hf_token=hf_token, cache_dir=cache_dir)
    image_generator = ImageGenerator(diffusion_client=diffusion_client)

    generation_results = image_generator.generate_images(
        image_prompts=image_prompts,
        num_inference_steps=10,
        guidance_scale=7.5,
        size=1024
    )

    print(f"\nSaving {len(generation_results)} images to: {RESULTS_DIR}")
    saved_metadata = []
    for res in generation_results:
        scene_num = res["scene_index"] + 1
        filename = f"scene_{scene_num:02d}.png"
        filepath = os.path.join(RESULTS_DIR, filename)
        res["image"].save(filepath)
        print(f"  → Saved: {filepath}")
        saved_metadata.append({
            "scene_index": res["scene_index"],
            "filename": filename,
            "prompt": res["prompt"]
        })

    save_json("pipeline_result.json", {"generated_images": saved_metadata})
    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    print("=" * 40)
    print("Running Full Comic Generation Pipeline")
    print("=" * 40)
    run_pipeline()