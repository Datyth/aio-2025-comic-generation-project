from PIL import Image, ImageDraw, ImageFont, ImageOps
import textwrap
import logging
from typing import List, Dict, Any
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from models.stable_diffusion import DiffusionClient

class ImageGenerator:
    def __init__(self, diffusion_client: DiffusionClient):
        self.diffusion_client = diffusion_client

    def add_caption_box(self, image: Image.Image, text: str) -> Image.Image:
        img_with_border = ImageOps.expand(image, border=15, fill="black")
        img_copy = img_with_border.convert("RGBA")
        overlay = Image.new("RGBA", img_copy.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        # Try to load Roboto-Regular from assets or current dir, fallback otherwise
        font_path = os.path.join(os.path.dirname(__file__), "..", "..", "Roboto-Regular.ttf")
        try:
            font = ImageFont.truetype(font_path, 32)
        except:
            try:
                font = ImageFont.truetype("Roboto-Regular.ttf", 32)
            except:
                font = ImageFont.load_default()

        wrapped_text = textwrap.wrap(text, width=55)
        line_height = 40
        text_total_height = len(wrapped_text) * line_height

        box_bottom = img_copy.size[1] - 30
        box_top = box_bottom - text_total_height - 30
        draw.rectangle(
            [(30, box_top), (img_copy.size[0]-30, box_bottom)],
            fill=(255, 255, 255, 235),
            outline=(0, 0, 0, 255),
            width=6
        )

        y_text = box_top + 15
        for line in wrapped_text:
            draw.text((45, y_text), line, font=font, fill="black")
            y_text += line_height

        return Image.alpha_composite(img_copy, overlay).convert("RGB")

    def generate_image(self, prompt: str, paragraph: str, num_inference_steps: int = 5, guidance_scale: float = 2.0, size: int = 1024) -> Image.Image:
        raw_img = self.diffusion_client.gen_image(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=size,
            height=size
        )
        if raw_img is not None:
            return self.add_caption_box(raw_img, paragraph)
        return None