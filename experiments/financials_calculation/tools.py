from pydantic import BaseModel, Field
from typing import Literal


class CostOfRevenueCalculationOutput(BaseModel):
    cost_of_revenue: float = Field(..., description="The extracted, calculated, or imputed cost of revenue value")
    method: Literal["direct_extraction", "calculation", "imputation"] = Field(..., description="Method used to determine cost of revenue")
    formula_used: str = Field(..., description="The specific formula or XBRL concept(s) used")
    reasoning: str = Field(..., description="Clear explanation of logic and assumptions made")
    confidence: Literal["High", "Medium", "Low"] = Field(..., description="Confidence level based on reliability of method used")


class FinancialsCalculationTool:
    @staticmethod
    def openai_tool_definition():
        return {
            "type": "function",
            "function": {
                "name": "cost_of_revenue_calculation",
                "description": "Extract or calculate the cost of revenue from XBRL facts.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cost_of_revenue": {
                            "type": "number",
                            "description": "The extracted, calculated, or imputed cost of revenue value"
                        },
                        "method": {
                            "type": "string",
                            "enum": ["direct_extraction", "calculation", "imputation"],
                            "description": "Method used to determine cost of revenue"
                        },
                        "formula_used": {
                            "type": "string",
                            "description": "The specific formula or XBRL concept(s) used"
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Clear explanation of logic and assumptions made"
                        },
                        "confidence": {
                            "type": "string",
                            "enum": ["High", "Medium", "Low"],
                            "description": "Confidence level based on reliability of method used"
                        }
                    },
                    "required": ["cost_of_revenue", "method", "formula_used", "reasoning", "confidence"],
                    "additionalProperties": False
                }
            }
        }
    
    @staticmethod
    def deepseek_tool_definition():
        return {
            "type": "function",
            "function": {
                "name": "cost_of_revenue_calculation",
                "description": "Extract or calculate the cost of revenue from XBRL facts.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cost_of_revenue": {
                            "type": "number",
                            "description": "The extracted, calculated, or imputed cost of revenue value"
                        },
                        "method": {
                            "type": "string",
                            "enum": ["direct_extraction", "calculation", "imputation"],
                            "description": "Method used to determine cost of revenue"
                        },
                        "formula_used": {
                            "type": "string",
                            "description": "The specific formula or XBRL concept(s) used"
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Clear explanation of logic and assumptions made"
                        },
                        "confidence": {
                            "type": "string",
                            "enum": ["High", "Medium", "Low"],
                            "description": "Confidence level based on reliability of method used"
                        }
                    },
                    "required": ["cost_of_revenue", "method", "formula_used", "reasoning", "confidence"],
                    "additionalProperties": False
                }
            }
        }
    
    @staticmethod
    def kimi_tool_definition():
        return {
            "type": "function",
            "function": {
                "name": "cost_of_revenue_calculation",
                "description": "Extract or calculate the cost of revenue from XBRL facts.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cost_of_revenue": {
                            "type": "number",
                            "description": "The extracted, calculated, or imputed cost of revenue value"
                        },
                        "method": {
                            "type": "string",
                            "enum": ["direct_extraction", "calculation", "imputation"],
                            "description": "Method used to determine cost of revenue"
                        },
                        "formula_used": {
                            "type": "string",
                            "description": "The specific formula or XBRL concept(s) used"
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Clear explanation of logic and assumptions made"
                        },
                        "confidence": {
                            "type": "string",
                            "enum": ["High", "Medium", "Low"],
                            "description": "Confidence level based on reliability of method used"
                        }
                    },
                    "required": ["cost_of_revenue", "method", "formula_used", "reasoning", "confidence"],
                    "additionalProperties": False
                }
            }
        }
    
    @staticmethod
    def anthropic_tool_definition():
        return {
        "name": "cost_of_revenue_calculation",
        "description": "Extract or calculate the cost of revenue from XBRL facts.",
        "input_schema": {
            "type": "object",
            "properties": {
                "cost_of_revenue": {
                    "type": "number",
                    "description": "The extracted, calculated, or imputed cost of revenue value"
                },
                "method": {
                    "type": "string",
                    "enum": ["direct_extraction", "calculation", "imputation"],
                    "description": "Method used to determine cost of revenue"
                },
                "formula_used": {
                    "type": "string",
                    "description": "The specific formula or XBRL concept(s) used"
                },
                "reasoning": {
                    "type": "string",
                    "description": "Clear explanation of logic and assumptions made"
                },
                "confidence": {
                    "type": "string",
                    "enum": ["High", "Medium", "Low"],
                    "description": "Confidence level based on reliability of method used"
                }
            },
            "required": ["cost_of_revenue", "method", "formula_used", "reasoning", "confidence"],
            "additionalProperties": False
        }
    }
    
    @staticmethod
    def gemini_tool_definition():
        return {
            "name": "cost_of_revenue_calculation",
            "description": "Extract or calculate the cost of revenue from XBRL facts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cost_of_revenue": {
                        "type": "number",
                        "description": "The extracted, calculated, or imputed cost of revenue value"
                    },
                    "method": {
                        "type": "string",
                        "enum": ["direct_extraction", "calculation", "imputation"],
                        "description": "Method used to determine cost of revenue"
                    },
                    "formula_used": {
                        "type": "string",
                        "description": "The specific formula or XBRL concept(s) used"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Clear explanation of logic and assumptions made"
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["High", "Medium", "Low"],
                        "description": "Confidence level based on reliability of method used"
                    }
                },
                "required": ["cost_of_revenue", "method", "formula_used", "reasoning", "confidence"]
            }
        }