"""
Green flag filter strategies.

This module contains filter implementations that identify companies
with positive financial characteristics.
"""

from .base import FilterStrategy


class GreenFlagFilter(FilterStrategy):
    """Filter for companies with strong financial performance indicators."""
    
    def get_filters(self) -> list[dict[str, str]]:
        return [
            { "field": "net_income", "operator": "gte", "value": 250000000 },
            { "field": "total_debt", "operator": "lt", "value": 2000000000 },
            { "field": "current_ratio", "operator": "gte", "value": 1.2 },
            { "field": "free_cash_flow", "operator": "gte", "value": 100000000 }
        ]

    
    def get_label(self) -> str:
        return "Green Flag" 