import os
import json
from typing import List, Dict, Any, Optional
import logging
from transformers import pipeline as text_pipeline
from dotenv import load_dotenv

from models.llm import LLMClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Scene:
    def __init__(self, description: str, characters: List[str], action: str, setting: str):
        self.description = description
        self.characters = characters
        self.action = action
        self.setting = setting

class Character:
    def __init__(self, name, appearance, personality, role):
        self.name = name
        self.appearance = appearance
        self.personality = personality
        self.role = role


class StoryGenerator:
    def __init__(self, llm: LLMClient):
        self.llm, self.tokenizer = llm.get_model()
        self.generator = self.prepare_generator()
        logger.info("Initialized Story Generator successfully.")

    def prepare_generator(self):
        return text_pipeline("text-generation", model=self.llm, tokenizer=self.tokenizer)

    def generate_text(self, prompt: str):
        output = self.generator(prompt, max_new_tokens=2048, return_full_text=False)
        return output[0]['generated_text']

    def gen_story(self, story: str, max_scenes: int = 3):
        scenes = self.decompose_story(story, max_scenes)
        characters = self.extract_characters(scenes)
        return scenes, characters

    def decompose_story(self, story: str, max_scenes: int = 3) -> List[Scene]:
        prompt = self.create_decompose_prompt(story, max_scenes)
        response = self.generate_text(prompt)
        scenes = self.parse_response(response)
        return scenes

    def create_decompose_prompt(self, story: str, max_scenes: int):
        prompt = f"""You are a comic book writer. Decompose the following story into {max_scenes} or fewer comic book scenes.

For each scene, provide:
1. "description": A brief description of what happens.
2. "characters": A list of characters present in the scene.
3. "action": Main action description.
4. "setting": Location description.

Story:
{story}

Format your response strictly as a JSON array of scenes. Each scene MUST have this exact structure:
[
  {{
    "description": "Brief scene description",
    "characters": ["character1", "character2"],
    "action": "Main action description",
    "setting": "Location description"
  }}
]
Respond with only the valid JSON array, no markdown formatting (like ```json), and no additional text."""
        return prompt

    def parse_response(self, response: str) -> List[Scene]:
        try:
            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            elif cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]

            cleaned_response = cleaned_response.strip()
            data = json.loads(cleaned_response)

            scenes = []
            for item in data:
                scene = Scene(
                    description=item.get("description", ""),
                    characters=item.get("characters", []),
                    action=item.get("action", ""),
                    setting=item.get("setting", "")
                )
                scenes.append(scene)
            return scenes
        except Exception as e:
            logger.error(f"Error parsing JSON: {e}")
            return []

    def extract_characters(self, scenes: List[Scene]):
        character_names = set()
        for scene in scenes:
            character_names.update(scene.characters)

        characters = []
        for character_name in character_names:
            character = self.generate_character_description(character_name, scenes)
            if character:
                characters.append(character)
        return characters

    def generate_character_description(self, character_name: str, scenes: List[Scene]):
        contexts = []
        for scene in scenes:
            if character_name in scene.characters:
                contexts.append(scene.description)
        context_str = "\n".join(contexts)

        prompt = f"""Based on these scenes, create a detailed character description for "{character_name}".
Context:
{context_str}

Provide:
1. Physical appearance (for visual consistency in comic generation)
2. Personality traits
3. Role in the story

Format your response as JSON:
{{
    "appearance": "Detailed physical description",
    "personality": "Personality traits",
    "role": "Character's role"
}}
Respond with only the valid JSON array, no markdown formatting, and no additional text."""

        try:
            response = self.generate_text(prompt)
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1:
                char_data = json.loads(response[start_idx:end_idx])
                return Character(
                    name=character_name,
                    appearance=char_data.get("appearance", ""),
                    personality=char_data.get("personality", ""),
                    role=char_data.get("role", "")
                )
        except Exception as e:
            logger.warning(f"Failed to generate character: {e}")
        return Character(character_name, f"Character named {character_name}", "", "")


if __name__ == "__main__":
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN", "")
    cache_dir = os.getenv("CACHE_DIR", "./cache")

    llm_client = LLMClient(hf_token=hf_token, cache_dir=cache_dir)
    story_generator = StoryGenerator(llm=llm_client)

    story_text = """
    Cậu bé vốn hay bỏ dở việc học, làm gì cũng chóng chán. Một hôm, cậu thấy một bà cụ kiên trì mài thỏi sắt thành kim, cậu ngạc nhiên và được bà nói rằng kiên trì mỗi ngày sẽ đạt kết quả. Bài học khiến cậu bé nhận ra giá trị của sự bền bỉ, từ đó chăm chỉ học tập và không còn bỏ dở như trước.
    """

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