from PIL import Image, ImageDraw, ImageFont
import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from app.models.stable_diffusion import DiffusionClient
from app.modules.image_prompt_generator import ImagePrompt, ImagePromptGenerator

class ImageGenerator:
    def __init__(self, diffusion_client: DiffusionClient):
        self.diffusion_client = diffusion_client
        logger.info("Initialized Image Generator with robust text wrapping successfully.")

    def generate_images(self, image_prompts: List[ImagePrompt], num_inference_steps: int = 50, guidance_scale: float = 7.5, size: int = 1024) -> List[Dict[str, Any]]:
        results = []

        for prompt_obj in image_prompts:
            logger.info(f"Generating image for Scene {prompt_obj.scene_index + 1}...")

            raw_img = self.diffusion_client.gen_image(
                prompt=prompt_obj.prompt,
                negative_prompt=prompt_obj.negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=size,
                height=size
            )
            final_img = self.add_caption_box(raw_img, prompt_obj.description)
            results.append({
                "scene_index": prompt_obj.scene_index,
                "image": final_img,
                "prompt": prompt_obj.prompt
            })

        logger.info(f"Finished generating {len(results)} images.")
        return results

    def add_caption_box(self, image: Image.Image, text: str, font_size: int = 40):
        img_copy = image.copy().convert("RGBA")
        width, height = img_copy.size
        overlay = Image.new("RGBA", img_copy.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()
            logger.warning("Custom font not found, using default. Text may be small.")

        safe_width = width - 120
        wrapped_text = self._wrap_text_by_pixels(text, font, draw, safe_width)

        bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font)
        text_height = bbox[3] - bbox[1]

        box_padding = 30
        box_height = text_height + (box_padding * 2)
        box_y_start = height - box_height

        draw.rectangle([(0, box_y_start), (width, height)], fill=(255, 255, 255, 220))

        text_x = (width - (bbox[2] - bbox[0])) / 2
        text_y = box_y_start + box_padding

        draw.multiline_text((text_x, text_y), wrapped_text, font=font, fill=(0, 0, 0, 255), align="center")

        return Image.alpha_composite(img_copy, overlay).convert("RGB")

    def _wrap_text_by_pixels(self, text, font, draw, max_width):
        if not text:
            return ""

        words = text.split(' ')
        lines = []
        current_line = []

        for word in words:
            test_line = ' '.join(current_line + [word])
            width = draw.textlength(test_line, font=font)

            if width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]

        if current_line:
            lines.append(' '.join(current_line))

        return '\n'.join(lines)

    def display_results(self, generation_results: List[Dict[str, Any]]):
        from IPython.display import display
        for res in generation_results:
            print(f"\n--- Screen {res['scene_index'] + 1} ---")
            display(res['image'])