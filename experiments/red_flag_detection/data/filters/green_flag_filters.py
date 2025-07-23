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
            {"field": "net_income", "operator": "gte", "value": 1000000000},
            {"field": "total_debt", "operator": "lt", "value": 500000000},
            {"field": "return_on_equity", "operator": "gte", "value": 15},
            {"field": "current_ratio", "operator": "gte", "value": 1.5},
            {"field": "free_cash_flow", "operator": "gte", "value": 500000000}
        ]
    
    def get_label(self) -> str:
        return "Green Flag" 