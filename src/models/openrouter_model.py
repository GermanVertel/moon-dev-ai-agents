"""
🌙 Moon Dev's OpenRouter Model Implementation
Built with love by Moon Dev 🚀

OpenRouter gives access to Claude, GPT, DeepSeek, Grok, Gemini, Llama and more
with a single API key through an OpenAI-compatible endpoint.
Any model slug from https://openrouter.ai/models works (format: "provider/model").
"""

from openai import OpenAI
from termcolor import cprint
from .base_model import BaseModel, ModelResponse

class OpenRouterModel(BaseModel):
    """Implementation for OpenRouter's models"""

    # Popular examples - any slug listed on openrouter.ai/models is accepted
    AVAILABLE_MODELS = {
        "openai/gpt-4o-mini": "Fast, cheap GPT-4o mini",
        "openai/gpt-4o": "GPT-4 Optimized",
        "anthropic/claude-3.5-haiku": "Fast Claude model",
        "anthropic/claude-sonnet-4": "Balanced Claude model",
        "deepseek/deepseek-chat": "DeepSeek chat model",
        "deepseek/deepseek-r1": "DeepSeek reasoning model",
        "x-ai/grok-4": "xAI Grok model",
        "google/gemini-2.0-flash-001": "Fast Gemini model",
        "meta-llama/llama-3.3-70b-instruct": "Llama 3.3 70B"
    }

    def __init__(self, api_key: str, model_name: str = "openai/gpt-4o-mini", base_url: str = "https://openrouter.ai/api/v1", **kwargs):
        self.model_name = model_name
        self.base_url = base_url
        super().__init__(api_key, **kwargs)

    def initialize_client(self, **kwargs) -> None:
        """Initialize the OpenRouter client"""
        try:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
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
        max_tokens: int = 1024,
        **kwargs
    ) -> ModelResponse:
        """Generate a response using OpenRouter"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False
            )

            return ModelResponse(
                content=(response.choices[0].message.content or "").strip(),
                raw_response=response,
                model_name=self.model_name,
                usage=response.usage.model_dump() if getattr(response, 'usage', None) else None
            )

        except Exception as e:
            cprint(f"❌ OpenRouter generation error: {str(e)}", "red")
            raise

    def is_available(self) -> bool:
        """Check if OpenRouter is available"""
        return self.client is not None

    @property
    def model_type(self) -> str:
        return "openrouter"
