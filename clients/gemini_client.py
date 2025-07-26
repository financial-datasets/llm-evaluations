from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class GeminiClient:
    """Utility class for accessing different Google Gemini LLM models."""
    
    def __init__(self, api_key: str | None = None):
        """Initialize the Gemini client."""
        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    
    def call(
        self, 
        model: str,
        messages: list[dict[str, str]], 
        max_tokens: int | None = None,
        temperature: float = 1.0,
        tools: list[dict[str, any]] | None = None,
        system: str | None = None
    ) -> any:
        """
        General method for calling any Gemini model.
        
        Args:
            model: Model name (e.g., "gemini-2.5-flash", "gemini-1.5-pro")
            messages: List of message dictionaries
            max_tokens: Maximum tokens to generate (None for model default)
            temperature: Temperature for randomness (0.0-2.0)
            tools: Optional list of tools for function calling
            system: Optional system prompt
            
        Returns:
            Gemini generate content response
        """
        # Convert messages to content format for Gemini
        contents = []
        
        # Add system message if provided
        if system:
            contents.append(system)
        
        # Convert messages to Gemini format
        for message in messages:
            if message["role"] == "user":
                contents.append(message["content"])
            elif message["role"] == "assistant":
                contents.append(message["content"])
        
        # If there's only one message, use it directly as string
        if len(contents) == 1:
            contents = contents[0]
        
        # Configure generation parameters
        config_params = {
            "temperature": temperature,
        }
        
        if max_tokens is not None:
            config_params["max_output_tokens"] = max_tokens
        
        # Add tools if provided
        if tools:
            # Convert tools to Gemini format
            function_declarations = []
            for tool in tools:
                function_declarations.append(tool)
            
            tool_config = types.Tool(function_declarations=function_declarations)
            config_params["tools"] = [tool_config]
        
        config = types.GenerateContentConfig(**config_params)
        
        return self.client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )


# Example usage
if __name__ == "__main__":
    client = GeminiClient()
    
    # Example with tools
    tools = [{
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
        },
    }]
    
    messages = [{"role": "user", "content": "What is the weather like in Paris today?"}]
    
    completion = client.call(
        model="gemini-2.5-flash",
        messages=messages,
        tools=tools
    )
    
    # Check for a function call
    if completion.candidates[0].content.parts[0].function_call:
        function_call = completion.candidates[0].content.parts[0].function_call
        print(f"Function to call: {function_call.name}")
        print(f"Arguments: {function_call.args}")
    else:
        print("No function call found in the response.")
        print(completion.text) 