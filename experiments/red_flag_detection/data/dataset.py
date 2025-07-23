"""
Red flag dataset container.

This module provides the RedFlagDetectionDataset class for organizing and accessing
financial data with different flag classifications.
"""

import json
import os
from typing import Optional


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
    
    def save_to_json(self, filepath: str) -> None:
        """Save the dataset to a JSON file."""
        # Ensure directory exists (only if filepath contains a directory)
        directory = os.path.dirname(filepath)
        if directory:  # Only create directory if filepath contains a path
            os.makedirs(directory, exist_ok=True)
        
        data = {
            'companies': self._companies,
            'metadata': {
                'total_companies': self.size(),
                'labels': list(self.labels()),
                'red_flag_count': len(self.get_red_flag_companies()),
                'green_flag_count': len(self.get_green_flag_companies())
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Dataset saved to {filepath}")
    
    @classmethod
    def load_from_json(cls, filepath: str) -> Optional['RedFlagDetectionDataset']:
        """Load dataset from a JSON file. Returns None if file doesn't exist."""
        if not os.path.exists(filepath):
            return None
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            companies = data.get('companies', [])
            print(f"Dataset loaded from {filepath} ({len(companies)} companies)")
            return cls(companies)
        
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error loading dataset from {filepath}: {e}")
            return None 