"""
Red flag dataset container.

This module provides the RedFlagDetectionDataset class for organizing and accessing
financial data with different flag classifications.
"""


class RedFlagDetectionDataset:
    """Dataset container for red flag detection data."""
    
    def __init__(self, companies: list[dict[str, str]]):
        self._companies = companies
    
    def get_companies(self) -> list[dict[str, str]]:
        """Get all companies in the dataset."""
        return self._companies
    
    def get_red_flag_companies(self) -> list[dict[str, str]]:
        """Get companies with red flag labels."""
        return [c for c in self._companies if c["label"] != "Green Flag"]
    
    def get_green_flag_companies(self) -> list[dict[str, str]]:
        """Get companies with green flag labels."""
        return [c for c in self._companies if c["label"] == "Green Flag"]
    
    def get_companies_by_label(self, label: str) -> list[dict[str, str]]:
        """Get companies with a specific label."""
        return [c for c in self._companies if c["label"] == label]
    
    def size(self) -> int:
        """Get the total number of companies in the dataset."""
        return len(self._companies)
    
    def labels(self) -> set[str]:
        """Get all unique labels in the dataset."""
        return {c["label"] for c in self._companies} 