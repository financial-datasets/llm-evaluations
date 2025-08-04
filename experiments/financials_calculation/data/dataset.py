"""
Financials calculation dataset container.

This module provides the FinancialsCalculationDataset class for organizing and accessing
financial data from SEC filings with XBRL facts for financial calculations.
"""

import json
import os
from typing import Optional


class FinancialsCalculationDataset:
    """Dataset container for financials calculation data."""
    
    def __init__(self, companies: list[dict]):
        self._companies = companies
    
    def get_companies(self) -> list[dict]:
        """Get all companies in the dataset."""
        return self._companies
    
    def get_companies_by_ticker(self, ticker: str) -> list[dict]:
        """Get companies with a specific ticker symbol."""
        return [c for c in self._companies if c.get("ticker") == ticker]
    
    def get_companies_by_filing_type(self, filing_type: str) -> list[dict]:
        """Get companies with a specific filing type (e.g., '10-Q', '10-K')."""
        return [c for c in self._companies if c.get("filing_type") == filing_type]
    
    def get_companies_by_report_period(self, report_period: str) -> list[dict]:
        """Get companies with a specific report period (e.g., '2025-06-30')."""
        return [c for c in self._companies if c.get("report_period") == report_period]
    
    def get_companies_with_xbrl_concept(self, concept: str) -> list[dict]:
        """Get companies that have a specific XBRL concept in their facts."""
        companies_with_concept = []
        for company in self._companies:
            xbrl_facts = company.get("xbrl_facts", [])
            if any(fact.get("concept") == concept for fact in xbrl_facts):
                companies_with_concept.append(company)
        return companies_with_concept
    
    def get_xbrl_facts_by_concept(self, concept: str) -> list[dict]:
        """Get all XBRL facts with a specific concept across all companies."""
        facts = []
        for company in self._companies:
            xbrl_facts = company.get("xbrl_facts", [])
            for fact in xbrl_facts:
                if fact.get("concept") == concept:
                    fact_with_company = fact.copy()
                    fact_with_company["ticker"] = company.get("ticker")
                    fact_with_company["cik"] = company.get("cik")
                    facts.append(fact_with_company)
        return facts
    
    def get_company_xbrl_facts(self, ticker: str = None, cik: str = None) -> list[dict]:
        """Get XBRL facts for a specific company by ticker or CIK."""
        if ticker:
            companies = self.get_companies_by_ticker(ticker)
        elif cik:
            companies = [c for c in self._companies if c.get("cik") == cik]
        else:
            return []
        
        if companies:
            return companies[0].get("xbrl_facts", [])
        return []
    
    def get_all_xbrl_concepts(self) -> set[str]:
        """Get all unique XBRL concepts in the dataset."""
        concepts = set()
        for company in self._companies:
            xbrl_facts = company.get("xbrl_facts", [])
            for fact in xbrl_facts:
                if "concept" in fact:
                    concepts.add(fact["concept"])
        return concepts
    
    def get_all_tickers(self) -> set[str]:
        """Get all unique ticker symbols in the dataset."""
        return {c.get("ticker") for c in self._companies if c.get("ticker")}
    
    def get_all_filing_types(self) -> set[str]:
        """Get all unique filing types in the dataset."""
        return {c.get("filing_type") for c in self._companies if c.get("filing_type")}
    
    def get_all_report_periods(self) -> set[str]:
        """Get all unique report periods in the dataset."""
        return {c.get("report_period") for c in self._companies if c.get("report_period")}
    
    def size(self) -> int:
        """Get the total number of companies in the dataset."""
        return len(self._companies)
    
    def total_xbrl_facts(self) -> int:
        """Get the total number of XBRL facts across all companies."""
        total = 0
        for company in self._companies:
            total += len(company.get("xbrl_facts", []))
        return total
    
    def save_to_json(self, filepath: str) -> None:
        """Save the dataset to a JSON file."""
        # Ensure directory exists (only if filepath contains a directory)
        directory = os.path.dirname(filepath)
        if directory:  # Only create directory if filepath contains a path
            os.makedirs(directory, exist_ok=True)
        
        data = self._companies
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Dataset saved to {filepath}")
    
    def get_metadata(self) -> dict:
        """Get metadata about the dataset."""
        return {
            'total_companies': self.size(),
            'total_xbrl_facts': self.total_xbrl_facts(),
            'unique_tickers': len(self.get_all_tickers()),
            'unique_filing_types': list(self.get_all_filing_types()),
            'unique_report_periods': len(self.get_all_report_periods()),
            'total_xbrl_concepts': len(self.get_all_xbrl_concepts())
        }
    
    @classmethod
    def load_from_json(cls, filepath: str) -> Optional['FinancialsCalculationDataset']:
        """Load dataset from a JSON file. Returns None if file doesn't exist."""
        if not os.path.exists(filepath):
            return None
        
        try:
            with open(filepath, 'r') as f:
                companies = json.load(f)
            
            print(f"Dataset loaded from {filepath} ({len(companies)} companies)")
            return cls(companies)
        
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error loading dataset from {filepath}: {e}")
            return None
