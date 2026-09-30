"""
🌙 Moon Dev's DeepSeek Model Implementation
Built with love by Moon Dev 🚀
"""

from openai import OpenAI
from termcolor import cprint
from .base_model import BaseModel, ModelResponse

class DeepSeekModel(BaseModel):
    """Implementation for DeepSeek's models"""
    
    AVAILABLE_MODELS = {
        "deepseek-chat": "DeepSeek-V3 general chat & coding model",
        "deepseek-reasoner": "DeepSeek-R1 enhanced reasoning model with thinking process",
        "deepseek-r1": "DeepSeek-R1 reasoning model (Alias for deepseek-reasoner)"
    }
    
    MODEL_ALIASES = {
        "deepseek-r1": "deepseek-reasoner",
        "deepseek-v3": "deepseek-chat"
    }
    
    def __init__(self, api_key: str, model_name: str = "deepseek-chat", base_url: str = "https://api.deepseek.com", **kwargs):
        self.model_name = model_name
        self.base_url = base_url
        super().__init__(api_key, **kwargs)
    
    def _get_api_model_name(self) -> str:
        """Resolve model alias to official API endpoint model name"""
        return self.MODEL_ALIASES.get(self.model_name, self.model_name)
    
    def initialize_client(self, **kwargs) -> None:
        """Initialize the DeepSeek client"""
        try:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            api_model = self._get_api_model_name()
            cprint(f"✨ Initialized DeepSeek model: {self.model_name} (API endpoint: {api_model})", "green")
        except Exception as e:
            cprint(f"❌ Failed to initialize DeepSeek model: {str(e)}", "red")
            self.client = None
    
    def generate_response(self, 
        system_prompt: str,
        user_content: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> ModelResponse:
        """Generate a response using DeepSeek with thinking/reasoning support"""
        try:
            api_model = self._get_api_model_name()
            
            request_params = {
                "model": api_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                "max_tokens": max_tokens,
                "stream": False,
                **kwargs
            }
            
            # DeepSeek Reasoner does not support custom temperature; only set for chat models
            if api_model != "deepseek-reasoner" and temperature is not None:
                request_params["temperature"] = temperature
            
            response = self.client.chat.completions.create(**request_params)
            
            message = response.choices[0].message
            content = (message.content or "").strip()
            reasoning_content = getattr(message, "reasoning_content", None)
            
            if reasoning_content:
                cprint(f"🧠 DeepSeek Reasoning process detected ({len(reasoning_content)} chars)", "magenta")
            
            return ModelResponse(
                content=content,
                raw_response=response,
                model_name=self.model_name,
                usage=response.usage.model_dump() if hasattr(response, 'usage') and response.usage else None,
                reasoning_content=reasoning_content
            )
            
        except Exception as e:
            cprint(f"❌ DeepSeek generation error: {str(e)}", "red")
            raise
    
    def is_available(self) -> bool:
        """Check if DeepSeek is available"""
        return self.client is not None
    
    @property
    def model_type(self) -> str:
        return "deepseek" 