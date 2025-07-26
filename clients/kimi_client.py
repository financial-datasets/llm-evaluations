import openai
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class KimiClient:
    """Utility class for accessing Kimi K2 models from MoonShot AI."""
    
    def __init__(self):
        """Initialize the Kimi client."""
        self.client = OpenAI(
            api_key=os.getenv("KIMI_API_KEY"),
            base_url="https://api.moonshot.ai/v1"
        )
    
    def call(
        self, 
        model: str = "kimi-k2-0711-preview",
        messages: list[dict[str, str]] = None, 
        max_tokens: int | None = None,
        temperature: float = 0.6,  # Recommended temperature for Kimi K2
        tools: list[dict[str, any]] | None = None,
        tool_choice: str | None = None,
        response_format: dict[str, str] | None = None,
        system: str | None = None
    ) -> openai.types.chat.ChatCompletion:
        """
        General method for calling Kimi K2 models.
        
        Args:
            model: Model name (defaults to "kimi-k2-instruct")
            messages: List of message dictionaries
            max_tokens: Maximum tokens to generate (None for model default)
            temperature: Temperature for randomness (0.6 recommended for Kimi K2)
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
    client = KimiClient()
    
    # Example with tools
    tools = [{
        "type": "function",
        "function": {
            "name": "CodeRunner",
            "description": "A code executor that supports running Python and JavaScript code",
            "parameters": {
                "type": "object",
                "properties": {
                    "language": {
                        "type": "string",
                        "enum": ["python", "javascript"],
                        "description": "The programming language to execute"
                    },
                    "code": {
                        "type": "string",
                        "description": "The code to execute"
                    }
                },
                "required": ["language", "code"],
                "additionalProperties": False
            }
        }
    }]
    
    messages = [
        {"role": "system", "content": "You are Kimi, an AI assistant created by Moonshot AI."},
        {"role": "user", "content": "Write a Python function to check if a number is prime."}
    ]
    
    completion = client.call(
        model="kimi-k2-0711-preview",
        messages=messages,
        tools=tools
    )
    
    print("Tool calls:", completion.choices[0].message.tool_calls)
