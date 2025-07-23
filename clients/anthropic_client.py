import anthropic
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AnthropicClient:
    """Utility class for accessing different Anthropic LLM models."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Anthropic client."""
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def call(
        self, 
        model: str,
        messages: List[Dict[str, str]], 
        max_tokens: int = 1024,
        temperature: float = 0.0,
        tools: Optional[List[Dict[str, Any]]] = None,
        system: Optional[str] = None
    ) -> anthropic.types.Message:
        """
        General method for calling any Anthropic model.
        
        Args:
            model: Model name (use Models class constants or string)
            messages: List of message dictionaries
            max_tokens: Maximum tokens to generate
            temperature: Temperature for randomness (0.0-1.0)
            tools: Optional list of tools for function calling
            system: Optional system prompt
            
        Returns:
            Anthropic message response
        """
        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        
        if tools:
            kwargs["tools"] = tools
        if system:
            kwargs["system"] = system
            
        return self.client.messages.create(**kwargs)

# Example usage
if __name__ == "__main__":
    client = AnthropicClient()
    
    # Example with tools
    tools = [{
        "name": "get_weather",
        "description": "Get current temperature for a given location.",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and country e.g. Bogotá, Colombia"
                }
            },
            "required": [
                "location"
            ]
        }
    }]
    
    messages = [{"role": "user", "content": "What is the weather like in Paris today?"}]
    
    completion = client.call(
        model="claude-3-haiku-20240307",
        messages=messages,
        tools=tools
    )
    
    print("Tool calls:", completion.content)