import re
import logging
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from models.llm import LLMClient

class ImagePromptGenerator:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def parse_paragraphs(self, raw_text: str) -> List[str]:
        paragraphs = []
        pattern = r"<\|paragraph\|>(.*?)(?=<\|paragraph\|>|<\|eos\|>|$)"
        for match in re.finditer(pattern, raw_text, re.DOTALL):
            content = match.group(1).strip()
            if content: 
                paragraphs.append(content)
        return paragraphs

    def create_visual_prompt(self, paragraph: str, full_story: str, style: str = "whimsical fable book illustration, highly detailed, vibrant colors, fantasy art style"):
        if self.llm_client is None:
            logger.warning("LLM Client does not exist. Return fallback prompt.")
            return f"{paragraph}, {style}"

        translation_prompt = f"""You are an expert art director. Given the full context of a Vietnamese story, translate the specific paragraph into a highly short descriptive English image prompt.
Maintain visual consistency of characters, objects, and environments based on the overall story.
Focus only on the visual elements (characters, actions, environment) of the specific paragraph.
Do NOT include any explanations or conversational text. Return ONLY the English translation, under 50 words.

Full Story Context: "{full_story}"

Specific Paragraph to visualize: "{paragraph}"

English Image Prompt:"""
        
        english_translation = self.llm_client.generate(translation_prompt, max_new_tokens=100)
        english_translation = english_translation.strip(' "\'\n')

        final_prompt = f"{english_translation}, {style}"
        return final_prompt
