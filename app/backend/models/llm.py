import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self, model_id="meta-llama/Meta-Llama-3-8B-Instruct", hf_token=None, cache_dir="./cache"):
        self.model_id = model_id
        self.hf_token = hf_token
        self.cache_dir = cache_dir
        self.model = None
        self.generator = None
        self.tokenizer = None

    def load_model(self):
        if self.model is not None:
            logger.info(f"LLM {self.model_id} already loaded. Skipping.")
            return

        logger.info(f"Loading LLM {self.model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            token=self.hf_token,
            cache_dir=self.cache_dir
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        llama3_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% if message['role'] == 'system' %}{{ '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}{% elif message['role'] == 'user' %}{{ '<|start_header_id|>user<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}{% elif message['role'] == 'assistant' %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
        self.tokenizer.chat_template = llama3_template

        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            quantization_config=bnb_config,
            token=self.hf_token,
            cache_dir=self.cache_dir
        )
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )
        logger.info("LLM loaded successfully.")

    def generate(self, prompt, max_new_tokens=512):
        if self.generator is None:
            self.load_model()

        messages = [
            {"role": "system", "content": "You are a helpful and precise assistant."},
            {"role": "user", "content": prompt}
        ]

        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        outputs = self.generator(
            prompt_text,
            max_new_tokens=max_new_tokens,
            return_full_text=False,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return outputs[0]['generated_text'].strip()