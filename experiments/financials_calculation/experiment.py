import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from clients.anthropic_client import AnthropicClient
from clients.openai_client import OpenAIClient
from clients.gemini_client import GeminiClient
from clients.kimi_client import KimiClient
from clients.deepseek_client import DeepSeekClient
from experiments.financials_calculation.data.dataset import FinancialsCalculationDataset
from experiments.financials_calculation.tools import CostOfRevenueCalculationOutput, FinancialsCalculationTool
from pydantic import BaseModel


class CostOfRevenuePredictionResult(BaseModel):
    """Single cost of revenue prediction result from an LLM."""
    ticker: str
    model: str
    prediction: float  # The predicted cost of revenue value
    ground_truth: float  # The actual cost of revenue from dataset
    reasoning: str
    method: str  # direct_extraction, calculation, or imputation
    formula_used: str  # The formula or concept used
    confidence: str  # High, Medium, or Low
    cost: float  # API call cost
    duration: float  # Time taken for the call


class ModelResults(BaseModel):
    """All cost of revenue prediction results from a specific model."""
    model_provider: str
    model_name: str
    predictions: list[CostOfRevenuePredictionResult]
    average_cost: float
    average_duration: float


class ExperimentResults(BaseModel):
    """Complete financials calculation experiment results from all models."""
    openai: ModelResults | None = None
    anthropic: ModelResults | None = None
    gemini: ModelResults | None = None
    kimi: ModelResults | None = None
    deepseek: ModelResults | None = None


class FinancialsCalculationExperiment:
  def __init__(self):
    self.anthropic_client = AnthropicClient()
    self.openai_client = OpenAIClient()
    self.gemini_client = GeminiClient()
    self.kimi_client = KimiClient()
    self.deepseek_client = DeepSeekClient()

  def run(self, dataset: FinancialsCalculationDataset) -> ExperimentResults:
    # Get the companies from the dataset
    companies = dataset.get_companies()

    # Execute all LLM calls in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all tasks
        future_to_provider = {
            executor.submit(self._call_openai, companies): "openai",
            executor.submit(self._call_anthropic, companies): "anthropic", 
            executor.submit(self._call_gemini, companies): "gemini",
            executor.submit(self._call_kimi, companies): "kimi",
            executor.submit(self._call_deepseek, companies): "deepseek"
        }
        
        # Collect results as they complete
        results = {}
        for future in as_completed(future_to_provider):
            provider = future_to_provider[future]
            try:
                result = future.result()
                results[provider] = result
                print(f"✅ {provider.capitalize()} completed")
            except Exception as e:
                print(f"❌ {provider.capitalize()} failed: {e}")
                results[provider] = None

    return ExperimentResults(
        openai=results.get("openai"),
        anthropic=results.get("anthropic"),
        gemini=results.get("gemini"),
        kimi=results.get("kimi"),
        deepseek=results.get("deepseek")
    )

  def _generate_prompt(self, ticker: str, xbrl_facts: list[dict]) -> list[dict]:
    """Format the user message for LLM input."""
    return [{
      "role": "user",
      "content": (
          f"You are a financial analyst. You are given XBRL facts from the income statement of the public company {ticker}.\n\n"
          f"Here are the XBRL facts:\n{json.dumps(xbrl_facts, indent=2)}\n\n"
          "Your job is to extract or calculate the **Cost of Revenue** for this company.\n\n"
          "**Instructions:**\n"
          "You must follow a strict hierarchy of approaches:\n\n"
          "### 1. **Direct Extraction**\n"
          "Look for any of the following XBRL concepts:\n"
          "- `us-gaap:CostOfRevenue`\n"
          "- `us-gaap:CostOfGoodsAndServicesSold`\n"
          "- `us-gaap:CostOfGoodsSold`\n"
          "- `us-gaap:CostOfServices`\n"
          "- `us-gaap:CostOfSales`\n"
          "If one of these is present, use its value directly.\n\n"
          "### 2. **Calculation-Based Estimation**\n"
          "If direct extraction is not possible, calculate using the first available formula below:\n"
          "- **Formula 1:** `us-gaap:Revenues` - `us-gaap:GrossProfit`\n"
          "- **Formula 2:** `us-gaap:CostOfGoodsSold` + `us-gaap:CostOfServices`\n"
          "- **Formula 3:** `us-gaap:OperatingExpenses` - `us-gaap:SellingGeneralAndAdministrativeExpense` - `us-gaap:ResearchAndDevelopmentExpense`\n"
          "- **Formula 4:** `us-gaap:CostOfRevenueFromContractWithCustomerExcludingAmortization` + `us-gaap:CostOfRevenueAmortization` + `us-gaap:CostOfRevenueHosting`\n"
          "- **Formula 5:** `us-gaap:CostOfSales`\n"
          "- **Formula 6:** `us-gaap:CostOfGoodsSold`\n\n"
          "### 3. **Imputation (Fallback Case)**\n"
          "If no formulas can be applied, and no direct tag is present, **impute** cost of revenue by using the following **industry-specific or ambiguous** tags when available:\n"
          "- `us-gaap:PolicyholderBenefitsAndClaimsIncurredNet`\n"
          "- `us-gaap:ClaimsAndClaimsAdjustmentExpenses`\n"
          "- `us-gaap:CostsAndExpenses`\n"
          "- `us-gaap:OperatingCostsAndExpenses`\n"
          "- `us-gaap:InterestExpenseBenefitNet`\n"
          "- `us-gaap:CostOfGoodsAndServicesSold` (if used in a non-standard context)\n"
          "Only use these tags if **none** of the above methods can be used.\n\n"
          "**Few-shot Examples:**\n\n"
          "**Example 1 - Direct Extraction:**\n"
          "```\n"
          '[{"concept": "us-gaap:CostOfRevenue", "numeric_value": 26932000}]\n'
          "Result: Cost of Revenue = 26,932,000 (directly extracted)\n"
          "```\n\n"
          "**Example 2 - Revenue minus Gross Profit:**\n"
          "```\n"
          '[{"concept": "us-gaap:Revenues", "numeric_value": 1615709000}, {"concept": "us-gaap:GrossProfit", "numeric_value": 341328000}]\n'
          "Result: Cost of Revenue = 1,615,709,000 - 341,328,000 = 1,274,381,000\n"
          "```\n\n"
          "**Example 3 - Imputed via Insurance Claim Costs:**\n"
          "```\n"
          '[{"concept": "us-gaap:PolicyholderBenefitsAndClaimsIncurredNet", "numeric_value": 1170000000}, {"concept": "us-gaap:PremiumsEarnedNet", "numeric_value": 1650000000}]\n'
          "Result: Cost of Revenue = 1,170,000,000 (imputed from PolicyholderBenefitsAndClaimsIncurredNet)\n"
          "```\n\n"
          "**Respond using the `cost_of_revenue_calculation` function call**, providing:\n"
          "- `cost_of_revenue`: The extracted, calculated, or imputed numeric value\n"
          "- `method`: One of 'direct_extraction', 'calculation', or 'imputation'\n"
          "- `formula_used`: The specific formula or concept(s) used\n"
          "- `reasoning`: Clear explanation of your logic and assumptions\n"
          "- `confidence`: High / Medium / Low based on the reliability of the method used"
      )
    }]



  def _call_openai(self, companies: list[dict]) -> ModelResults:
    predictions = []
    model = "o3"
    input_cost_per_million_tokens = 2.50
    output_cost_per_million_tokens = 10.00
    total_cost = 0.0
    total_time = 0.0

    
    print(f"🤖 OpenAI: extracting financials for {len(companies)} companies...")

    for i, company in enumerate(companies, 1):
        # Add sleep to avoid rate limiting
        time.sleep(1)

        ticker = company["ticker"]
        xbrl_facts = company["xbrl_facts"]
        print(f"  OpenAI: processing {ticker} ({i}/{len(companies)})")

        messages = self._generate_prompt(ticker, xbrl_facts)

        try:
            # Start timing the API call
            start_time = time.time()
            
            response = self.openai_client.call(
              model=model,
              messages=messages,
              tools=[FinancialsCalculationTool.openai_tool_definition()],
              tool_choice={"type": "function", "function": {"name": "cost_of_revenue_calculation"}}
            )
            
            # End timing the API call
            end_time = time.time()
            call_duration = end_time - start_time
            total_time += call_duration

            # Calculate cost from usage
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            
            input_cost = (prompt_tokens / 1_000_000) * input_cost_per_million_tokens
            output_cost = (completion_tokens / 1_000_000) * output_cost_per_million_tokens
            call_cost = input_cost + output_cost
            total_cost += call_cost

            # Parse the tool call
            tool_calls = response.choices[0].message.tool_calls
            if not tool_calls:
                print(f"No tool call returned for {ticker}")
                continue

            args = json.loads(tool_calls[0].function.arguments)
            parsed = CostOfRevenueCalculationOutput(**args)

            predictions.append(CostOfRevenuePredictionResult(
                ticker=ticker,
                model=model,
                ground_truth=company.get("cost_of_revenue"),
                prediction=parsed.cost_of_revenue,
                reasoning=parsed.reasoning,
                method=parsed.method,
                formula_used=parsed.formula_used,
                confidence=parsed.confidence,
                cost=call_cost,
                duration=call_duration,
            ))

            # Add sleep to avoid rate limiting
            time.sleep(1)

        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    return ModelResults(
        model_provider="openai", 
        model_name=model, 
        predictions=predictions, 
        average_cost=total_cost / len(predictions) if predictions else 0,
        average_duration=total_time / len(predictions) if predictions else 0
    )
  
  def _call_anthropic(self, companies: list[dict]) -> ModelResults:
    predictions = []
    model = "claude-opus-4-20250514"
    input_cost_per_million_tokens = 3.00
    output_cost_per_million_tokens = 15.00
    total_cost = 0.0
    total_time = 0.0
    
    print(f"🧠 Anthropic: extracting financials for {len(companies)} companies...")

    for i, company in enumerate(companies, 1):
        # Add sleep to avoid rate limiting
        time.sleep(1)

        ticker = company["ticker"]
        xbrl_facts = company["xbrl_facts"]
        print(f"  Anthropic: processing {ticker} ({i}/{len(companies)})")

        messages = self._generate_prompt(ticker, xbrl_facts)

        try:
            # Start timing the API call
            start_time = time.time()
            
            response = self.anthropic_client.call(
                model=model,
                messages=messages,
                tools=[FinancialsCalculationTool.anthropic_tool_definition()]
            )
            
            # End timing the API call
            end_time = time.time()
            call_duration = end_time - start_time
            total_time += call_duration

            # Calculate cost from usage
            prompt_tokens = response.usage.input_tokens
            completion_tokens = response.usage.output_tokens
            
            input_cost = (prompt_tokens / 1_000_000) * input_cost_per_million_tokens
            output_cost = (completion_tokens / 1_000_000) * output_cost_per_million_tokens
            call_cost = input_cost + output_cost
            total_cost += call_cost

            # Find the tool use block in the response content
            tool_use = None
            for content_block in response.content:
                if content_block.type == "tool_use":
                    tool_use = content_block
                    break
            
            if not tool_use:
                print(f"No tool call returned for {ticker}")
                continue

            args = tool_use.input
            parsed = CostOfRevenueCalculationOutput(**args)

            predictions.append(CostOfRevenuePredictionResult(
                ticker=ticker,
                model=model,
                ground_truth=company.get("cost_of_revenue"),
                prediction=parsed.cost_of_revenue,
                reasoning=parsed.reasoning,
                method=parsed.method,
                formula_used=parsed.formula_used,
                confidence=parsed.confidence,
                cost=call_cost,
                duration=call_duration,
            ))

            # Add sleep to avoid rate limiting
            time.sleep(1)

        except Exception as e:
            print(f"Error processing {ticker} with Claude: {e}")

    return ModelResults(
        model_provider="anthropic", 
        model_name=model, 
        predictions=predictions,
        average_cost=total_cost / len(predictions) if predictions else 0,
        average_duration=total_time / len(predictions) if predictions else 0
    )
  
  def _call_gemini(self, companies: list[dict]) -> ModelResults:
    predictions = []
    model = "gemini-2.5-pro"
    input_cost_per_million_tokens = 2.50
    output_cost_per_million_tokens = 10.00
    total_cost = 0.0
    total_time = 0.0
    
    print(f"💎 Gemini: extracting financials for {len(companies)} companies...")

    for i, company in enumerate(companies, 1):
        # Add sleep to avoid rate limiting
        time.sleep(1)

        ticker = company["ticker"]
        xbrl_facts = company["xbrl_facts"]
        print(f"  Gemini: processing {ticker} ({i}/{len(companies)})")

        messages = self._generate_prompt(ticker, xbrl_facts)

        try:
            # Start timing the API call
            start_time = time.time()
            
            response = self.gemini_client.call(
                model=model,
                messages=messages,
                tools=[FinancialsCalculationTool.gemini_tool_definition()]
            )
            
            # End timing the API call
            end_time = time.time()
            call_duration = end_time - start_time
            total_time += call_duration

            # Calculate cost from usage
            prompt_tokens = response.usage_metadata.prompt_token_count
            completion_tokens = response.usage_metadata.candidates_token_count
            
            input_cost = (prompt_tokens / 1_000_000) * input_cost_per_million_tokens
            output_cost = (completion_tokens / 1_000_000) * output_cost_per_million_tokens
            call_cost = input_cost + output_cost
            total_cost += call_cost

            # Check for function call in Gemini response
            function_call = None
            if (response.candidates and 
                response.candidates[0].content.parts and 
                response.candidates[0].content.parts[0].function_call):
                function_call = response.candidates[0].content.parts[0].function_call
            
            if not function_call:
                print(f"No tool call returned for {ticker}")
                continue

            # Get arguments from function call
            args = function_call.args
            parsed = CostOfRevenueCalculationOutput(**args)

            predictions.append(CostOfRevenuePredictionResult(
                ticker=ticker,
                model=model,
                ground_truth=company.get("cost_of_revenue"),
                prediction=parsed.cost_of_revenue,
                reasoning=parsed.reasoning,
                method=parsed.method,
                formula_used=parsed.formula_used,
                confidence=parsed.confidence,
                cost=call_cost,
                duration=call_duration,
            ))

            # Add sleep to avoid rate limiting
            time.sleep(1)

        except Exception as e:
            print(f"Error processing {ticker} with Gemini: {e}")

    return ModelResults(
        model_provider="gemini", 
        model_name=model, 
        predictions=predictions,
        average_cost=total_cost / len(predictions) if predictions else 0,
        average_duration=total_time / len(predictions) if predictions else 0
    )

  def _call_kimi(self, companies: list[dict]) -> ModelResults:
    predictions = []
    model = "kimi-k2-0711-preview"
    input_cost_per_million_tokens = 1.00  # Estimated pricing
    output_cost_per_million_tokens = 3.00  # Estimated pricing
    total_cost = 0.0
    total_time = 0.0
    
    print(f"🌙 Kimi: extracting financials for {len(companies)} companies...")

    for i, company in enumerate(companies, 1):      
        # Add sleep to avoid rate limiting
        time.sleep(1)

        ticker = company["ticker"]
        xbrl_facts = company["xbrl_facts"]
        print(f"  Kimi: processing {ticker} ({i}/{len(companies)})")

        messages = self._generate_prompt(ticker, xbrl_facts)

        try:
            # Start timing the API call
            start_time = time.time()
            
            response = self.kimi_client.call(
                model=model,
                messages=messages,
                tools=[FinancialsCalculationTool.kimi_tool_definition()],
                tool_choice={"type": "function", "function": {"name": "cost_of_revenue_calculation"}}
            )
            
            # End timing the API call
            end_time = time.time()
            call_duration = end_time - start_time
            total_time += call_duration

            # Calculate cost from usage
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            
            input_cost = (prompt_tokens / 1_000_000) * input_cost_per_million_tokens
            output_cost = (completion_tokens / 1_000_000) * output_cost_per_million_tokens
            call_cost = input_cost + output_cost
            total_cost += call_cost

            # Parse the tool call (OpenAI-style response)
            tool_calls = response.choices[0].message.tool_calls
            if not tool_calls:
                print(f"No tool call returned for {ticker}")
                continue

            args = json.loads(tool_calls[0].function.arguments)
            parsed = CostOfRevenueCalculationOutput(**args)

            predictions.append(CostOfRevenuePredictionResult(
                ticker=ticker,
                model=model,
                ground_truth=company.get("cost_of_revenue"),
                prediction=parsed.cost_of_revenue,
                reasoning=parsed.reasoning,
                method=parsed.method,
                formula_used=parsed.formula_used,
                confidence=parsed.confidence,
                cost=call_cost,
                duration=call_duration,
            ))

        except Exception as e:
            print(f"Error processing {ticker} with Kimi: {e}")

    return ModelResults(
        model_provider="kimi", 
        model_name=model, 
        predictions=predictions,
        average_cost=total_cost / len(predictions) if predictions else 0,
        average_duration=total_time / len(predictions) if predictions else 0
    )

  def _call_deepseek(self, companies: list[dict]) -> ModelResults:
    predictions = []
    model = "deepseek-reasoner"
    input_cost_per_million_tokens = 0.14  # Estimated pricing based on DeepSeek's competitive rates
    output_cost_per_million_tokens = 0.28  # Estimated pricing
    total_cost = 0.0
    total_time = 0.0
    
    print(f"🔍 DeepSeek: extracting financials for {len(companies)} companies...")

    for i, company in enumerate(companies, 1):
        # Add sleep to avoid rate limiting
        time.sleep(1)

        ticker = company["ticker"]
        xbrl_facts = company["xbrl_facts"]
        print(f"  DeepSeek: processing {ticker} ({i}/{len(companies)})")

        messages = self._generate_prompt(ticker, xbrl_facts)

        try:
            # Start timing the API call
            start_time = time.time()
            
            response = self.deepseek_client.call(
                model=model,
                messages=messages,
                tools=[FinancialsCalculationTool.deepseek_tool_definition()]
            )
            
            # End timing the API call
            end_time = time.time()
            call_duration = end_time - start_time
            total_time += call_duration

            # Calculate cost from usage
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            
            input_cost = (prompt_tokens / 1_000_000) * input_cost_per_million_tokens
            output_cost = (completion_tokens / 1_000_000) * output_cost_per_million_tokens
            call_cost = input_cost + output_cost
            total_cost += call_cost

            # Parse the tool call (OpenAI-style response)
            tool_calls = response.choices[0].message.tool_calls
            if not tool_calls:
                print(f"No tool call returned for {ticker}")
                continue

            args = json.loads(tool_calls[0].function.arguments)
            parsed = CostOfRevenueCalculationOutput(**args)

            predictions.append(CostOfRevenuePredictionResult(
                ticker=ticker,
                model=model,
                ground_truth=company.get("cost_of_revenue"),
                prediction=parsed.cost_of_revenue,
                reasoning=parsed.reasoning,
                method=parsed.method,
                formula_used=parsed.formula_used,
                confidence=parsed.confidence,
                cost=call_cost,
                duration=call_duration,
            ))

            # Add sleep to avoid rate limiting
            time.sleep(1)

        except Exception as e:
            print(f"Error processing {ticker} with DeepSeek: {e}")

    return ModelResults(
        model_provider="deepseek", 
        model_name=model, 
        predictions=predictions,
        average_cost=total_cost / len(predictions) if predictions else 0,
        average_duration=total_time / len(predictions) if predictions else 0
    )


