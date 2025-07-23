"""
Red flag filter strategies.

This module contains filter implementations that identify companies
with concerning financial characteristics.
"""

from .base import FilterStrategy


class FinancialHealthIssuesFilter(FilterStrategy):
    """Filter for companies with poor financial health metrics."""
    
    def get_filters(self) -> list[dict[str, str]]:
        return [
            {"field": "current_ratio", "operator": "lt", "value": 1.0},
            {"field": "quick_ratio", "operator": "lt", "value": 0.8},
            {"field": "debt_to_equity", "operator": "gt", "value": 2.0},
            {"field": "total_debt", "operator": "gt", "value": 2000000000}
        ]
    
    def get_label(self) -> str:
        return "Financial Health Issues"


class DecliningProfitabilityFilter(FilterStrategy):
    """Filter for companies with declining profitability."""
    
    def get_filters(self) -> list[dict[str, str]]:
        return [
            {"field": "net_margin", "operator": "lt", "value": 5.0},
            {"field": "operating_margin", "operator": "lt", "value": 5.0},
            {"field": "net_income", "operator": "lt", "value": 0}
        ]
    
    def get_label(self) -> str:
        return "Declining Profitability"


class EarningsDeclineFilter(FilterStrategy):
    """Filter for companies with declining earnings and growth metrics."""
    
    def get_filters(self) -> list[dict[str, str]]:
        return [
            {"field": "earnings_growth", "operator": "lt", "value": 0},
            {"field": "free_cash_flow_growth", "operator": "lt", "value": 0},
            {"field": "revenue_growth", "operator": "lt", "value": 0}
        ]
    
    def get_label(self) -> str:
        return "Earnings Decline"


class BankruptcyRiskFilter(FilterStrategy):
    """Filter for companies with high bankruptcy risk indicators."""
    
    def get_filters(self) -> list[dict[str, str]]:
        return [
            {"field": "operating_cash_flow", "operator": "lt", "value": 0},
            {"field": "interest_coverage", "operator": "lt", "value": 1.5},
            {"field": "cash_ratio", "operator": "lt", "value": 0.5}
        ]
    
    def get_label(self) -> str:
        return "Bankruptcy Risk"


class InefficientOperationsFilter(FilterStrategy):
    """Filter for companies with inefficient operational metrics."""
    
    def get_filters(self) -> list[dict[str, str]]:
        return [
            {"field": "inventory_turnover", "operator": "lt", "value": 2.0},
            {"field": "receivables_turnover", "operator": "lt", "value": 4.0},
            {"field": "asset_turnover", "operator": "lt", "value": 0.5}
        ]
    
    def get_label(self) -> str:
        return "Inefficient Operations" 