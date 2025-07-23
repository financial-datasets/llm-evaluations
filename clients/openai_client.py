import openai
from openai import OpenAI
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class OpenAIClient:
    """Utility class for accessing different OpenAI LLM models."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the OpenAI client."""
        self.client = OpenAI(api_key=api_key)
    
    def call(
        self, 
        model: str,
        messages: List[Dict[str, str]], 
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        response_format: Optional[Dict[str, str]] = None,
        system: Optional[str] = None
    ) -> openai.types.chat.ChatCompletion:
        """
        General method for calling any OpenAI model.
        
        Args:
            model: Model name (e.g., "gpt-4", "gpt-3.5-turbo")
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
    client = OpenAIClient()
    
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
        model="gpt-4.1",
        messages=messages,
        tools=tools
    )
    
    print("Tool calls:", completion.choices[0].message.tool_calls)

