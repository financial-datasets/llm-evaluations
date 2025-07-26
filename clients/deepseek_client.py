import openai
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DeepSeekClient:
    """Utility class for accessing DeepSeek LLM models."""
    
    def __init__(self):
        """Initialize the DeepSeek client."""
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
    
    def call(
        self, 
        model: str = "deepseek-reasoner",
        messages: list[dict[str, str]] = None, 
        max_tokens: int | None = None,
        temperature: float = 0.0,
        tools: list[dict[str, any]] | None = None,
        tool_choice: str | None = None,
        response_format: dict[str, str] | None = None,
        system: str | None = None
    ) -> openai.types.chat.ChatCompletion:
        """
        General method for calling DeepSeek models.
        
        Args:
            model: Model name (defaults to "deepseek-chat")
            messages: List of message dictionaries
            max_tokens: Maximum tokens to generate (None for model default)
            temperature: Temperature for randomness (0.0-2.0)
            tools: Optional list of tools for function calling
            tool_choice: Optional tool choice ("auto", "none", or specific tool)
            response_format: Optional response format (e.g., {"type": "json_object"})
            system: Optional system prompt (will be prepended to messages)
            
        Returns:
            OpenAI chat completion response
        """
        if messages is None:
            messages = []
            
        # Add system message if provided
        if system:
            messages = [{"role": "system", "content": system}] + messages
        
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
        if response_format:
            kwargs["response_format"] = response_format
            
        return self.client.chat.completions.create(**kwargs)


# Example usage
if __name__ == "__main__":
    client = DeepSeekClient()
    
    # Example with tools
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current temperature for a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and country e.g. Bogotá, Colombia"
                    }
                },
                "required": [
                    "location"
                ],
                "additionalProperties": False
            },
            "strict": True
        }
    }]
    
    messages = [{"role": "user", "content": "What is the weather like in Paris today?"}]
    
    completion = client.call(
        model="deepseek-chat",
        messages=messages,
        tools=tools
    )
    
    print("Tool calls:", completion.choices[0].message.tool_calls)