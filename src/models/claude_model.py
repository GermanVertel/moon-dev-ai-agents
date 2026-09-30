"""
🌙 Moon Dev's Claude Model Implementation
Built with love by Moon Dev 🚀
"""

from anthropic import Anthropic
from termcolor import cprint
from .base_model import BaseModel, ModelResponse

class ClaudeModel(BaseModel):
    """Implementation for Anthropic's Claude models"""
    
    AVAILABLE_MODELS = {
        "claude-3-7-sonnet-latest": "Most intelligent Claude model with hybrid reasoning/thinking capabilities",
        "claude-3-5-sonnet-latest": "Industry-leading intelligence and speed for coding & reasoning",
        "claude-3-5-haiku-latest": "Fastest and most cost-effective Claude model",
        "claude-3-opus-latest": "Deep complex analysis model",
        "claude-3-haiku-20240307": "Legacy fast model"
    }
    
    MODEL_ALIASES = {
        "claude-3.7-sonnet": "claude-3-7-sonnet-latest",
        "claude-3-7-sonnet": "claude-3-7-sonnet-latest",
        "claude-3.5-sonnet": "claude-3-5-sonnet-latest",
        "claude-3-5-sonnet": "claude-3-5-sonnet-latest",
        "claude-3.5-haiku": "claude-3-5-haiku-latest",
        "claude-3-5-haiku": "claude-3-5-haiku-latest",
        "claude-3-opus": "claude-3-opus-latest",
        "claude-3-sonnet": "claude-3-5-sonnet-latest",
        "claude-3-haiku": "claude-3-5-haiku-latest"
    }
    
    def __init__(self, api_key: str, model_name: str = "claude-3-5-haiku-latest", **kwargs):
        self.model_name = model_name
        super().__init__(api_key, **kwargs)
    
    def _get_api_model_name(self) -> str:
        """Resolve model alias to official API endpoint model name"""
        return self.MODEL_ALIASES.get(self.model_name, self.model_name)
    
    def initialize_client(self, **kwargs) -> None:
        """Initialize the Anthropic client"""
        try:
            self.client = Anthropic(api_key=self.api_key)
            api_model = self._get_api_model_name()
            cprint(f"✨ Initialized Claude model: {self.model_name} (API endpoint: {api_model})", "green")
        except Exception as e:
            cprint(f"❌ Failed to initialize Claude model: {str(e)}", "red")
            self.client = None
    
    def generate_response(self, 
        system_prompt: str,
        user_content: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> ModelResponse:
        """Generate a response using Claude with thinking/reasoning support"""
        try:
            api_model = self._get_api_model_name()
            
            request_params = {
                "model": api_model,
                "max_tokens": max_tokens,
                "system": system_prompt,
                "messages": [
                    {"role": "user", "content": user_content}
                ],
                **kwargs
            }
            
            # Anthropic requires temperature=1.0 or omitted if extended thinking is active
            if "thinking" in kwargs:
                request_params["temperature"] = 1.0
            elif temperature is not None:
                request_params["temperature"] = temperature
                
            response = self.client.messages.create(**request_params)
            
            # Parse response blocks (text & thinking)
            text_blocks = []
            thinking_blocks = []
            
            for block in response.content:
                block_type = getattr(block, "type", None)
                if block_type == "text" or hasattr(block, "text"):
                    text_blocks.append(block.text)
                elif block_type == "thinking" or hasattr(block, "thinking"):
                    thinking_blocks.append(block.thinking)
            
            content = "".join(text_blocks).strip()
            reasoning_content = "".join(thinking_blocks).strip() if thinking_blocks else None
            
            if reasoning_content:
                cprint(f"🧠 Claude Extended Thinking detected ({len(reasoning_content)} chars)", "magenta")
            
            # Extract detailed usage
            usage = None
            if hasattr(response, "usage") and response.usage:
                usage = {
                    "prompt_tokens": getattr(response.usage, "input_tokens", 0),
                    "completion_tokens": getattr(response.usage, "output_tokens", 0),
                    "total_tokens": getattr(response.usage, "input_tokens", 0) + getattr(response.usage, "output_tokens", 0),
                    "cache_creation_input_tokens": getattr(response.usage, "cache_creation_input_tokens", 0),
                    "cache_read_input_tokens": getattr(response.usage, "cache_read_input_tokens", 0)
                }
            
            return ModelResponse(
                content=content,
                raw_response=response,
                model_name=self.model_name,
                usage=usage,
                reasoning_content=reasoning_content
            )
            
        except Exception as e:
            cprint(f"❌ Claude generation error: {str(e)}", "red")
            raise
    
    def is_available(self) -> bool:
        """Check if Claude is available"""
        return self.client is not None
    
    @property
    def model_type(self) -> str:
        return "claude" 