import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(
        self,
        hf_token: str,
        model_id: str = "Qwen/Qwen2.5-3B-Instruct",
        cache_dir: Optional[str] = "./cache"
    ):
        self.model_id = model_id
        self.hf_token = hf_token
        self.cache_dir = cache_dir
        logger.info(f"Loading model '{model_id}' (cache_dir='{cache_dir}')...")
        self.llm, self.tokenizer = self.prepare_llm()

    def prepare_llm(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            token=self.hf_token,
            cache_dir=self.cache_dir
        )
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        llm_model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            token=self.hf_token,
            cache_dir=self.cache_dir
        )
        logger.info(f"Model loaded successfully from '{self.cache_dir}'.")
        return llm_model, tokenizer

    def get_model(self):
        """Return the (model, tokenizer) tuple."""
        return self.llm, self.tokenizer