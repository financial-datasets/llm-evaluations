import requests
import json
import os
# Load environment variables
from dotenv import load_dotenv


load_dotenv()

class FinancialDatasetsClient:

  def __init__(self):
    self._base_url = "https://api.financialdatasets.ai"
    self._headers = {"X-API-KEY": os.getenv('FINANCIAL_DATASETS_API_KEY')}


  def search(self, filters: list[dict], label: str, limit: int = 5, period: str = "ttm") -> list[dict[str, str]]:
    url = f"{self._base_url}/financials/search"
    body = {
        "period": period,
        "limit": limit,
        "filters": filters
    }
    response = requests.request("POST", url, json=body, headers=self._headers)
    response.raise_for_status()
    results = response.json().get("search_results", [])
    return [{"ticker": r["ticker"], "label": label} for r in results] 