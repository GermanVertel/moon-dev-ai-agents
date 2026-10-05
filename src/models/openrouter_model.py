"""
🌙 Moon Dev's OpenRouter Model Implementation
Built with love by Moon Dev 🚀

OpenRouter gives access to Claude, GPT, DeepSeek, Grok, Gemini, Llama and more
with a single API key through an OpenAI-compatible endpoint.
Any model slug from https://openrouter.ai/models works (format: "provider/model").
"""

import os
import time
from openai import OpenAI
from termcolor import cprint
from .base_model import BaseModel, ModelResponse

class OpenRouterModel(BaseModel):
    """Implementation for OpenRouter's models"""

    # Popular examples - any slug listed on openrouter.ai/models is accepted
    AVAILABLE_MODELS = {
        "openai/gpt-4o-mini": "Fast, cheap GPT-4o mini",
        "openai/gpt-4o": "GPT-4 Optimized",
        "openai/o3-mini": "OpenAI Reasoning Model",
        "anthropic/claude-3.5-haiku": "Fast Claude model",
        "anthropic/claude-3.5-sonnet": "Balanced Claude model",
        "anthropic/claude-3.7-sonnet": "Advanced Hybrid Claude model",
        "deepseek/deepseek-chat": "DeepSeek chat model",
        "deepseek/deepseek-r1": "DeepSeek reasoning model",
        "x-ai/grok-4": "xAI Grok model",
        "google/gemini-2.0-flash-001": "Fast Gemini model",
        "meta-llama/llama-3.3-70b-instruct": "Llama 3.3 70B"
    }

    def __init__(self, api_key: str = None, model_name: str = "openai/gpt-4o", base_url: str = "https://openrouter.ai/api/v1", **kwargs):
        api_key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_KEY")
        self.model_name = model_name
        self.base_url = base_url
        super().__init__(api_key, **kwargs)

    def initialize_client(self, **kwargs) -> None:
        """Initialize the OpenRouter client with timeout and retries"""
        try:
            if not self.api_key:
                cprint("❌ OpenRouter API key not found in environment", "red")
                self.client = None
                return

            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=120.0,
                max_retries=3,
                default_headers={"X-Title": "Moon Dev AI Agents"}  # Shows up in OpenRouter dashboard
            )
            cprint(f"✨ Initialized OpenRouter model: {self.model_name}", "green")
        except Exception as e:
            cprint(f"❌ Failed to initialize OpenRouter model: {str(e)}", "red")
            self.client = None

    def generate_response(self,
        system_prompt: str,
        user_content: str,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        **kwargs
    ) -> ModelResponse:
        """Generate a response using OpenRouter with retry backoff"""
        if not self.client:
            raise RuntimeError("OpenRouter client not initialized")

        timeout = kwargs.get("timeout", 120)
        max_attempts = kwargs.get("max_retries", 3)
        last_err = None

        for attempt in range(1, max_attempts + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    stream=False
                )

                return ModelResponse(
                    content=(response.choices[0].message.content or "").strip(),
                    raw_response=response,
                    model_name=self.model_name,
                    usage=response.usage.model_dump() if getattr(response, 'usage', None) else None
                )

            except Exception as e:
                last_err = e
                err_msg = str(e)
                cprint(f"⚠️ OpenRouter attempt {attempt}/{max_attempts} failed: {err_msg[:120]}", "yellow")
                if attempt < max_attempts:
                    time.sleep(2 ** attempt)

        cprint(f"❌ OpenRouter generation error after {max_attempts} attempts: {last_err}", "red")
        raise last_err

    def is_available(self) -> bool:
        """Check if OpenRouter is available"""
        return self.client is not None

    @property
    def model_type(self) -> str:
        return "openrouter"
