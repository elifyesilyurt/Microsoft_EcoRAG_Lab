import json
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class ExtractedMetric(BaseModel):
    entity: str = Field(description="The subject/entity (e.g., Replenishment Projects, Water Benefit, Zero Waste Datacenters)")
    metric_type: str = Field(description="Type of metric: volume, count, emissions, percentage, energy")
    value: float = Field(description="The numeric value extracted")
    unit: str = Field(description="Unit of measurement: m3, million m3, projects, metric tons, datacenters, %")
    temporal_scope: str = Field(description="Time period: FY20, FY23, FY24, FY25, Lifetime")
    is_cumulative: bool = Field(description="True if lifetime/cumulative total, False if annual/in-year")
    raw_quote: str = Field(description="Exact sentence quote from context")

class QueryExtractionPlan(BaseModel):
    reasoning: str
    metrics: List[ExtractedMetric]
    information_found: bool

EXTRACTION_SYSTEM_PROMPT = """You are a Strict Quantitative Information Extraction Engine.
Your task is to extract all relevant numerical metrics, physical units, entities, and temporal scopes from the context matching the user query.

STRICT PROTOCOL:
1. Extract numerical values with their EXACT bound units (e.g., m3, million m3, projects, metric tons).
2. Differentiate strictly between counts (e.g., projects, facilities) and physical metrics (e.g., m3, metric tons).
3. Differentiate strictly between cumulative/lifetime totals and single-year additions.
4. If the exact specific information requested is completely missing, set \"information_found\": false and \"metrics\": [].

Respond ONLY with a valid JSON object adhering strictly to the JSON schema:
{
  \"reasoning\": \"brief explanation\",
  \"information_found\": true,
  \"metrics\": [
    {
      \"entity\": \"string\",
      \"metric_type\": \"volume|count|emissions|percentage|energy\",
      \"value\": 0.0,
      \"unit\": \"string\",
      \"temporal_scope\": \"string\",
      \"is_cumulative\": true,
      \"raw_quote\": \"string\"
    }
  ]
}"""

def format_extraction_prompt(query: str, context_chunks: List[str]) -> str:
    joined_context = "\n---\n".join(context_chunks)
    return f"Context:\n{joined_context}\n\nUser Query: {query}\n\nOutput JSON:"

class DeterministicResolver:
    @staticmethod
    def validate_and_filter(extraction_plan: QueryExtractionPlan, query: str) -> Dict[str, Any]:
        if not extraction_plan.information_found or not extraction_plan.metrics:
            return {
                "status": "NOT_FOUND",
                "message": "The provided documents do not contain the specific data requested."
            }

        query_lower = query.lower()
        wants_volume = any(w in query_lower for w in ["volume", "consumed", "withdrawn", "hacim", "benefit"])
        wants_count = any(w in query_lower for w in ["how many", "count", "number of", "projects count"])

        validated_results = []
        for m in extraction_plan.metrics:
            # Soru yalnızca hacim istiyorken modelin getirdiği proje/bina adedini programatik olarak filtrele
            if wants_volume and not wants_count and m.metric_type == "count" and m.unit in ["projects", "buildings", "facilities"]:
                continue
            validated_results.append(m)

        if not validated_results:
            return {
                "status": "NOT_FOUND",
                "message": "No matching volumetric metrics found after assertion checks."
            }

        return {
            "status": "SUCCESS",
            "metrics": validated_results
        }
