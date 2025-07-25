from pydantic import BaseModel, Field


class RedFlagDetectionOutput(BaseModel):
    has_red_flags: bool = Field(..., description="True if the company has red flags, False otherwise")
    reasoning: str = Field(..., description="Explanation of the decision, citing relevant financial metrics")


class RedFlagDetectionTool:
    @staticmethod
    def openai_tool_definition():
        return {
            "type": "function",
            "function": {
                "name": "red_flag_detection",
                "description": "Determine if a company has financial red flags based on its financial metrics.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "has_red_flags": {
                            "type": "boolean",
                            "description": "True if the company has financial red flags"
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Explanation for the red flag judgment, referencing financial metrics"
                        }
                    },
                    "required": ["has_red_flags", "reasoning"],
                    "additionalProperties": False
                }
            }
        }
    
    @staticmethod
    def anthropic_tool_definition():
        return {
        "name": "red_flag_detection",
        "description": "Determine if a company has financial red flags based on its financial metrics.",
        "input_schema": {
            "type": "object",
            "properties": {
                "has_red_flags": {
                    "type": "boolean",
                    "description": "True if the company has financial red flags"
                },
                "reasoning": {
                    "type": "string",
                    "description": "Explanation for the red flag judgment, referencing financial metrics"
                }
            },
            "required": ["has_red_flags", "reasoning"],
            "additionalProperties": False
        }
    }
    
    @staticmethod
    def gemini_tool_definition():
        return {
            "name": "red_flag_detection",
            "description": "Determine if a company has financial red flags based on its financial metrics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "has_red_flags": {
                        "type": "boolean",
                        "description": "True if the company has financial red flags"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Explanation for the red flag judgment, referencing financial metrics"
                    }
                },
                "required": ["has_red_flags", "reasoning"]
            }
        }