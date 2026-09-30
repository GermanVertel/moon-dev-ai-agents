# 🌙 Moon Dev's Model Factory

A unified interface for managing multiple AI model providers. This module handles initialization, API key management, and provides a consistent interface for generating responses across different AI models.

## 🔑 Required API Keys

Add these to your `.env` file in the project root:
```env
ANTHROPIC_KEY=your_key_here    # For Claude models
GROQ_API_KEY=your_key_here     # For Groq models (includes Mixtral, Llama, etc.)
OPENAI_KEY=your_key_here       # For OpenAI models (GPT-4, O1, etc.)
GEMINI_KEY=your_key_here       # For Gemini models
DEEPSEEK_KEY=your_key_here     # For DeepSeek models
OPENROUTER_API_KEY=your_key_here  # For OpenRouter (one key for many providers)
```

## 🤖 Available Models

### OpenAI Models
Latest Models:
- `gpt-5`: Next-generation GPT model (use when you want the strongest reasoning + code generation)
- `gpt-4o`: Latest GPT-4 Optimized model (Best for complex reasoning)
- `gpt-4o-mini`: Smaller, faster GPT-4 Optimized model (Good balance of speed/quality)
- `o1`: Latest O1 model (Dec 2024) - Shows reasoning process
- `o1-mini`: Smaller O1 model - Shows reasoning process
- `o3-mini`: Brand new fast reasoning model

### Claude Models (Anthropic)
Latest Models:
- `claude-3-7-sonnet-latest`: Most intelligent Claude model with hybrid reasoning/extended thinking
- `claude-3-5-sonnet-latest`: Industry-leading intelligence and speed for coding & reasoning
- `claude-3-5-haiku-latest`: Fastest, most cost-effective Claude model
- `claude-3-opus-latest`: Deep complex analysis model
- `claude-3-haiku-20240307`: Legacy fast model

### Gemini Models (Google)
Latest Models:
- `gemini-2.0-flash-exp`: Next-gen multimodal model (Audio, images, video, text)
- `gemini-2.0-flash`: Fast, efficient model optimized for quick responses
- `gemini-1.5-flash`: Fast versatile model (Audio, images, video, text)
- `gemini-1.5-flash-8b`: High volume tasks (Audio, images, video, text)
- `gemini-1.5-pro`: Complex reasoning tasks (Audio, images, video, text)
- `gemini-1.0-pro`: Natural language & code (Deprecated 2/15/2025)
- `text-embedding-004`: Text embeddings model

### Groq Models
Production Models:
- `mixtral-8x7b-32768`: Mixtral 8x7B (32k context) - $0.27/1M tokens
- `gemma2-9b-it`: Google Gemma 2 9B (8k context) - $0.10/1M tokens
- `llama-3.3-70b-versatile`: Llama 3.3 70B (128k context) - $0.70/1M in, $0.90/1M out
- `llama-3.1-8b-instant`: Llama 3.1 8B (128k context) - $0.10/1M tokens
- `llama-guard-3-8b`: Llama Guard 3 8B (8k context) - $0.20/1M tokens
- `llama3-70b-8192`: Llama 3 70B (8k context) - $0.70/1M in, $0.90/1M out
- `llama3-8b-8192`: Llama 3 8B (8k context) - $0.10/1M tokens

Preview Models:
- `deepseek-r1-distill-llama-70b`: DeepSeek R1 (128k context) - Shows thinking process
- `llama-3.3-70b-specdec`: Llama 3.3 70B SpecDec (8k context)
- `llama-3.2-1b-preview`: Llama 3.2 1B (128k context)
- `llama-3.2-3b-preview`: Llama 3.2 3B (128k context)

### DeepSeek Models
- `deepseek-chat`: DeepSeek-V3 general chat & coding model (Fast & cost-effective)
- `deepseek-reasoner`: DeepSeek-R1 enhanced reasoning model with thinking process (Chain-of-Thought)
- `deepseek-r1`: DeepSeek-R1 alias (automatically routes to `deepseek-reasoner`)

### OpenRouter Models
One `OPENROUTER_API_KEY` gives access to models from many providers. Use any slug from https://openrouter.ai/models (format `provider/model`), e.g.:
- `openai/gpt-4o-mini`: Default - fast and cheap
- `anthropic/claude-sonnet-4`: Balanced Claude model
- `deepseek/deepseek-r1`: DeepSeek reasoning model
- `x-ai/grok-4`: xAI Grok model

```python
model = model_factory.get_model("openrouter", "anthropic/claude-sonnet-4")
```

### Local Ollama: Free, Fast, Private LLMs 🚀

To get started with Ollama:
1. Install Ollama: `curl https://ollama.ai/install.sh | sh`
2. Start the server: `ollama serve`
3. Pull our models:
   ```bash
   ollama pull deepseek-r1      # DeepSeek R1 7B - shows thinking process
   ollama pull gemma:2b         # Google's Gemma 2B - fast responses
   ollama pull llama3.2         # Meta's Llama 3.2 - balanced performance
   ```
4. Check they're ready: `ollama list`

Available Models:
- `deepseek-r1`: Good for complex reasoning (7B parameters), shows thinking process with <think> tags
- `gemma:2b`: Fast and efficient for simple tasks, great for high-volume processing
- `llama3.2`: Balanced model good for most tasks, especially good at following instructions

Benefits:
- 🚀 Free to use - no API costs
- 🔒 Private - runs 100% local
- ⚡ Fast responses
- 🤔 DeepSeek shows thinking process
- 🛠️ Full model control

Usage Example:
```python
from src.models import model_factory

# Initialize with Llama 3.2 for balanced performance
model = factory.get_model("ollama", "llama3.2")

# Or use DeepSeek R1 for complex reasoning
model = factory.get_model("ollama", "deepseek-r1")

# Or Gemma for faster responses
model = factory.get_model("ollama", "gemma:2b")

# For the most powerful reasoning, use DeepSeek API
model = factory.get_model("deepseek", "deepseek-reasoner")
```

Interesting models for future use:
- gemma - for quick llm tasks https://huggingface.co/google/gemma-2-9b
- coqui - for voice locally https://huggingface.co/coqui/XTTS-v2

## 🚀 Usage Example

```python
from src.models import model_factory

# Initialize the model factory
factory = model_factory.ModelFactory()

# Get a specific model
model = factory.get_model("openai", "gpt-4o")  # Using latest GPT-4 Optimized

# Generate a response
response = model.generate_response(
    system_prompt="You are a helpful AI assistant.",
    user_content="Hello!",
    temperature=0.7,  # Optional: Control randomness (0.0-1.0)
    max_tokens=1024   # Optional: Control response length
)

print(response.content)
```

## 🌟 Features
- Unified interface for multiple AI providers
- Automatic API key validation and error handling
- Detailed debugging output with emojis
- Easy model switching with consistent interface
- Consistent response format across all providers
- Automatic handling of model-specific features:
  - Reasoning process display (O1, DeepSeek R1)
  - Context window management
  - Token counting and limits
  - Error recovery and retries

## 🔄 Model Updates
New models are regularly added to the factory. Check the Moon Dev Discord or GitHub for announcements about new models and features.

## 🐛 Troubleshooting
- If a model fails to initialize, check your API key in the `.env` file
- Some models (O1, DeepSeek R1) show their thinking process - this is normal
- For rate limit errors, try using a different model or wait a few minutes
- Watch Moon Dev's streams for live debugging and updates: [@moondevonyt](https://www.youtube.com/@moondevonyt)

## 🤝 Contributing
Feel free to contribute new models or improvements! Join the Moon Dev community:
- YouTube: [@moondevonyt](https://www.youtube.com/@moondevonyt)
- GitHub: [moon-dev-ai-agents-for-trading](https://github.com/moon-dev-ai-agents-for-trading)

Built with 💖 by Moon Dev 🌙
