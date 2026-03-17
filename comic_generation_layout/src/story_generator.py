import logging
from llm import LLMClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StoryGenerator:
    def __init__(self, llm_client: LLMClient):
        self.model = llm_client

    def load(self):
        logger.info("Initializing Story Generator...")
        self.model.load_model()

    def create_story_prompt(self, user_story: str) -> str:
        return f"""Hãy mở rộng đoạn tóm tắt dưới đây thành một câu chuyện ngụ ngôn.
Tóm tắt câu chuyện:
{user_story}

Bạn phải định dạng đầu ra chính xác như sau, nội dung từng đoạn phải bắt đầu bằng một thẻ <|paragraph|>:
<|paragraph|> [Nội dung đoạn 1] <|paragraph|> [Nội dung đoạn 2]

Chú ý:
- Không sinh thêm thẻ nào ngoài các thẻ <|paragraph|> đã được quy định.
- Phong cách kể chuyện ngụ ngôn, không cần sinh tên cụ thể cho nhân vật.
- Nội dung sinh ra là Tiếng Việt, giới hạn dưới 20 từ mỗi đoạn.
"""

    def gen_story_structured(self, user_story: str, max_new_tokens: int = 400) -> str:
        final_prompt = self.create_story_prompt(user_story)
        return self.model.generate(final_prompt, max_new_tokens=max_new_tokens)