import json
from typing import List
import logging
from transformers import pipeline as text_pipeline

from app.models.llm import LLMClient
from app.modules.story_generator import Scene, Character

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImagePrompt:
    def __init__(self, prompt: str, character_context: str, negative_prompt: str, scene_index: int = 0, description: str = ""):
        self.prompt = prompt
        self.char_contexts = character_context
        self.negative_prompt = negative_prompt
        self.scene_index = scene_index
        self.description = description

class ImagePromptGenerator:
    def __init__(self, llm_client: LLMClient):
        self.llm, self.tokenizer = llm_client.get_model()
        self.generator = self.prepare_generator()
        logger.info("Initialized Image Prompt Generator successfully.")

    def prepare_generator(self):
            return text_pipeline("text-generation", model=self.llm, tokenizer=self.tokenizer)

    def generate_text(self, prompt: str):
        output = self.generator(prompt, max_new_tokens=1024, return_full_text=False)
        return output[0]['generated_text']

    def generate_prompts(self, scenes: List[Scene], characters: List[Character], style: str = "art book style"):
        char_lookup = {char.name: char.appearance for char in characters}
        prompts = []

        for i, scene in enumerate(scenes):
                prompt = self.generate_scene_prompt(scene, i, char_lookup, style)
                prompts.append(prompt)

        logger.info(f"Generated {len(prompts)} image prompts")
        return prompts

    def generate_scene_prompt(self, scene: Scene, scene_index: int, char_lookup: dict, style: str = "art book style"):
        character_descriptions = []
        for char_name in scene.characters:
            appearance = char_lookup.get(char_name, f"a character named {char_name}")
            character_descriptions.append(f"{char_name}: {appearance}")

        char_context_str = " | ".join(character_descriptions)

        prompt = (
            f"{style}, Scene {scene_index + 1}: {scene.description}. "
            f"Location: {scene.setting}. "
            f"Action: {scene.action}. "
            f"Character visual details: {char_context_str}."
        )

        enhanced_prompt = self.enhance_prompt(prompt, scene, style)
        negative_prompt = self.create_negative_prompt()
        return ImagePrompt(
            prompt = enhanced_prompt,
            character_context = char_context_str,
            negative_prompt = negative_prompt,
            description = scene.description,
            scene_index = scene_index
        )

    def enhance_prompt(self, base_prompt: str, scene: Scene, style: str = "art book style"):
        enhanced_prompt = f"""You are an expert at creating image generation prompts for comic books.
Scene Description: {scene.description}
Setting: {scene.setting}
Action: {scene.action}
Characters: {', '.join(scene.characters)}

Base prompt: {base_prompt}
Style: {style}

Create a detailed, vivid image generation prompt that:
1. Describes the visual composition
2. Captures the mood and atmosphere
3. Specifies camera angle and framing
4. Includes the style "{style}"
5. Is optimized for image generation AI (be specific about visual elements)

Keep it under 100 words. Respond with only the prompt, no additional text."""

        try:
            enhanced_text = self.generate_text(enhanced_prompt)
            return enhanced_text.strip().replace('\n', ', ')
        except Exception as e:
            logger.warning(f"Enhancement failed, using base prompt: {e}")
            return base_prompt

    def create_negative_prompt(self):
        return (
                "extra characters, additional people, background crowd, "
                "more than the specified characters, multiple figures, "
                "extra animals, blurry, low quality, distorted, deformed, "
                "watermark, text, speech bubbles"
            )

