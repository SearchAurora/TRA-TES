"""
LLM Client — QwenMax via DashScope OpenAI-compatible endpoint
"""
import time
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

QWEN_API_KEY = "sk-e289af135f7a4c138aa16c11f69cfd59"
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL    = "qwen-max"


class LLMClient:
    def __init__(self, max_retries: int = 3, temperature: float = 0.7, max_tokens: int = 2048):
        self.max_retries  = max_retries
        self.temperature  = temperature
        self.max_tokens   = max_tokens
        self._client = OpenAI(api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)

    def complete(self, system: str, user: str) -> str:
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=QWEN_MODEL,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                wait = 2 ** attempt
                logger.warning(f"LLM attempt {attempt+1} failed: {e}. Retry in {wait}s")
                time.sleep(wait)
        raise RuntimeError(f"LLM failed after {self.max_retries} retries")
