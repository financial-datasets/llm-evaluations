import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel
from typing import Optional
from clients.anthropic_client import AnthropicClient
from clients.openai_client import OpenAIClient
from clients.gemini_client import GeminiClient
from clients.kimi_client import KimiClient
from clients.deepseek_client import DeepSeekClient
from experiments.red_flag_detection.data.dataset import RedFlagDetectionDataset
from experiments.red_flag_detection.data.factory import create_dataset
from experiments.red_flag_detection.tools import RedFlagDetectionOutput, RedFlagDetectionTool


class LLMPredictionResult(BaseModel):
    """Single prediction result from an LLM."""
    ticker: str
    model: str
    prediction: bool
    ground_truth: bool
    ground_truth_label: str
    reasoning: str
    cost: float
    duration: float

class ModelResults(BaseModel):
    """All results from a specific model."""
    model_provider: str
    model_name: str
    predictions: list[LLMPredictionResult]
    average_cost: float
    average_duration: float


class ExperimentResults(BaseModel):
    """Complete experiment results from all models."""
    openai: Optional[ModelResults] = None
    anthropic: Optional[ModelResults] = None
    gemini: Optional[ModelResults] = None
    kimi: Optional[ModelResults] = None
    deepseek: Optional[ModelResults] = None

class RedFlagDetectionExperiment:
  def __init__(self):
    self.anthropic_client = AnthropicClient()
    self.openai_client = OpenAIClient()
    self.gemini_client = GeminiClient()
    self.kimi_client = KimiClient()
    self.deepseek_client = DeepSeekClient()

  def run(self, dataset: RedFlagDetectionDataset) -> ExperimentResults:
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

  def _generate_prompt(self, ticker: str, metrics: dict) -> list[dict]:
    """Format the user message for LLM input."""
    return [{
        "role": "user",
        "content": (
            f"You are a financial analyst. You are given the financial metrics for the public company {ticker}.\n\n"
            f"Here are the financial metrics:\n{json.dumps(metrics, indent=2)}\n\n"
            "Your job is to determine whether this company shows signs of financial red flags.\n\n"
            "**Respond using the red_flag_detection function call**, with:\n"
            "- `has_red_flags: true` if the company appears financially risky (e.g., negative cash flow, high debt, poor liquidity, declining earnings).\n"
            "- `has_red_flags: false` if the company appears financially healthy overall.\n"
            "Also include a short explanation citing relevant metrics."
        )
    }]

  def _call_openai(self, companies: list[dict]) -> ModelResults:
    predictions = []
    model = "o3"
    input_cost_per_million_tokens = 2.50
    output_cost_per_million_tokens = 10.00
    total_cost = 0.0
    total_time = 0.0

    
    print(f"\n🤖 OpenAI: Processing {len(companies)} companies...")

    for i, company in enumerate(companies, 1):
        ticker = company["ticker"]
        metrics = company["financial_metrics"]
        print(f"  OpenAI ({i}/{len(companies)}): {ticker}")

        messages = self._generate_prompt(ticker, metrics)

        try:
            # Start timing the API call
            start_time = time.time()
            
            response = self.openai_client.call(
              model=model,
              messages=messages,
              tools=[RedFlagDetectionTool.openai_tool_definition()],
              tool_choice={"type": "function", "function": {"name": "red_flag_detection"}}
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
            parsed = RedFlagDetectionOutput(**args)

            predictions.append(LLMPredictionResult(
                ticker=ticker,
                model=model,
                ground_truth=company.get("label") != "Green Flag",
                ground_truth_label=company.get("label"),
                prediction=parsed.has_red_flags,
                reasoning=parsed.reasoning,
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
    
    print(f"\n🧠 Anthropic: Processing {len(companies)} companies...")

    for i, company in enumerate(companies, 1):
        ticker = company["ticker"]
        metrics = company["financial_metrics"]
        print(f"  Anthropic ({i}/{len(companies)}): {ticker}")

        messages = self._generate_prompt(ticker, metrics)

        try:
            # Start timing the API call
            start_time = time.time()
            
            response = self.anthropic_client.call(
                model=model,
                messages=messages,
                tools=[RedFlagDetectionTool.anthropic_tool_definition()]
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
            parsed = RedFlagDetectionOutput(**args)

            predictions.append(LLMPredictionResult(
                ticker=ticker,
                model=model,
                ground_truth=company.get("label") != "Green Flag",
                ground_truth_label=company.get("label"),
                prediction=parsed.has_red_flags,
                reasoning=parsed.reasoning,
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
    
    print(f"\n💎 Gemini: Processing {len(companies)} companies...")

    for i, company in enumerate(companies, 1):
        ticker = company["ticker"]
        metrics = company["financial_metrics"]
        print(f"  Gemini ({i}/{len(companies)}): {ticker}")

        messages = self._generate_prompt(ticker, metrics)

        try:
            # Start timing the API call
            start_time = time.time()
            
            response = self.gemini_client.call(
                model=model,
                messages=messages,
                tools=[RedFlagDetectionTool.gemini_tool_definition()]
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
            parsed = RedFlagDetectionOutput(**args)

            predictions.append(LLMPredictionResult(
                ticker=ticker,
                model=model,
                ground_truth=company.get("label") != "Green Flag",
                ground_truth_label=company.get("label"),
                prediction=parsed.has_red_flags,
                reasoning=parsed.reasoning,
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
    
    print(f"\n🌙 Kimi: Processing {len(companies)} companies...")

    for i, company in enumerate(companies, 1):
        ticker = company["ticker"]
        metrics = company["financial_metrics"]
        print(f"  Kimi ({i}/{len(companies)}): {ticker}")

        messages = self._generate_prompt(ticker, metrics)

        try:
            # Start timing the API call
            start_time = time.time()
            
            response = self.kimi_client.call(
                model=model,
                messages=messages,
                tools=[RedFlagDetectionTool.kimi_tool_definition()],
                tool_choice={"type": "function", "function": {"name": "red_flag_detection"}}
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
            parsed = RedFlagDetectionOutput(**args)

            predictions.append(LLMPredictionResult(
                ticker=ticker,
                model=model,
                ground_truth=company.get("label") != "Green Flag",
                ground_truth_label=company.get("label"),
                prediction=parsed.has_red_flags,
                reasoning=parsed.reasoning,
                cost=call_cost,
                duration=call_duration,
            ))

            # Add sleep to avoid rate limiting
            time.sleep(1)

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
    
    print(f"\n🔍 DeepSeek: Processing {len(companies)} companies...")

    for i, company in enumerate(companies, 1):
        ticker = company["ticker"]
        metrics = company["financial_metrics"]
        print(f"  DeepSeek ({i}/{len(companies)}): {ticker}")

        messages = self._generate_prompt(ticker, metrics)

        try:
            # Start timing the API call
            start_time = time.time()
            
            response = self.deepseek_client.call(
                model=model,
                messages=messages,
                tools=[RedFlagDetectionTool.deepseek_tool_definition()]
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
            parsed = RedFlagDetectionOutput(**args)

            predictions.append(LLMPredictionResult(
                ticker=ticker,
                model=model,
                ground_truth=company.get("label") != "Green Flag",
                ground_truth_label=company.get("label"),
                prediction=parsed.has_red_flags,
                reasoning=parsed.reasoning,
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


