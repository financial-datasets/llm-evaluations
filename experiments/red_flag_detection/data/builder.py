"""
Dataset builder for red flag detection.

This module provides the RedFlagDetectionDatasetBuilder class for constructing
datasets with configurable filter strategies.
"""

from typing import Optional
from clients.fd_client import FinancialDatasetsClient
from .dataset import RedFlagDetectionDataset
from .filters import (
    FilterStrategy,
    FinancialHealthIssuesFilter,
    DecliningProfitabilityFilter,
    EarningsDeclineFilter,
    BankruptcyRiskFilter,
    InefficientOperationsFilter,
    GreenFlagFilter,
)


class RedFlagDetectionDatasetBuilder:
    """Builder for constructing RedFlagDetectionDataset with configurable filters."""
    
    def __init__(self, fd_client: Optional[FinancialDatasetsClient] = None):
        self._fd_client = fd_client or FinancialDatasetsClient()
        self._filter_strategies: list[FilterStrategy] = []
        self._limit = 5
        self._period = "ttm"
    
    def with_financial_health_issues(self) -> 'RedFlagDetectionDatasetBuilder':
        """Add financial health issues filter."""
        self._filter_strategies.append(FinancialHealthIssuesFilter())
        return self
    
    def with_declining_profitability(self) -> 'RedFlagDetectionDatasetBuilder':
        """Add declining profitability filter."""
        self._filter_strategies.append(DecliningProfitabilityFilter())
        return self
    
    def with_earnings_decline(self) -> 'RedFlagDetectionDatasetBuilder':
        """Add earnings decline filter."""
        self._filter_strategies.append(EarningsDeclineFilter())
        return self
    
    def with_bankruptcy_risk(self) -> 'RedFlagDetectionDatasetBuilder':
        """Add bankruptcy risk filter."""
        self._filter_strategies.append(BankruptcyRiskFilter())
        return self
    
    def with_inefficient_operations(self) -> 'RedFlagDetectionDatasetBuilder':
        """Add inefficient operations filter."""
        self._filter_strategies.append(InefficientOperationsFilter())
        return self
    
    def with_green_flags(self) -> 'RedFlagDetectionDatasetBuilder':
        """Add green flag filter."""
        self._filter_strategies.append(GreenFlagFilter())
        return self
    
    def with_all_red_flags(self) -> 'RedFlagDetectionDatasetBuilder':
        """Add all red flag filters."""
        return (self.with_financial_health_issues()
                   .with_declining_profitability()
                   .with_earnings_decline()
                   .with_bankruptcy_risk()
                   .with_inefficient_operations())
    
    def with_custom_filter(self, filter_strategy: FilterStrategy) -> 'RedFlagDetectionDatasetBuilder':
        """Add a custom filter strategy."""
        self._filter_strategies.append(filter_strategy)
        return self
    
    def with_limit(self, limit: int) -> 'RedFlagDetectionDatasetBuilder':
        """Set the limit for search results per filter."""
        self._limit = limit
        return self
    
    def with_period(self, period: str) -> 'RedFlagDetectionDatasetBuilder':
        """Set the period for financial data (e.g., 'ttm', 'annual')."""
        self._period = period
        return self
    
    def build(self) -> RedFlagDetectionDataset:
        """Build and return the dataset."""
        if not self._filter_strategies:
            raise ValueError("At least one filter strategy must be added before building the dataset")
        
        all_companies = []
        for strategy in self._filter_strategies:
            companies = self._fd_client.search(
                filters=strategy.get_filters(),
                label=strategy.get_label(),
                limit=self._limit,
                period=self._period
            )
            all_companies.extend(companies)
        
        return RedFlagDetectionDataset(all_companies) 