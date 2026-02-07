from pydantic import BaseModel, Field
from typing import List
from datetime import datetime

class Quantiles(BaseModel):
    """
    The probabilistic forecast payload.
    Contains three distinct scenarios (Optimistic, Median, Pessimistic).
    """
    p10: List[float] = Field(..., description="10th percentile (Low carbon scenario)")
    p50: List[float] = Field(..., description="50th percentile (Median/Most likely)")
    p90: List[float] = Field(..., description="90th percentile (High carbon scenario)")

class ForecastResponse(BaseModel):
    """
    The main response object for /predict.
    Includes metadata for traceability (version, timestamp).
    """
    model_version: str = Field(..., description="The version tag of the model used (e.g. 'v2.0')")
    generated_at: datetime = Field(..., description="UTC timestamp of inference execution")
    horizon: int = Field(..., description="Number of hours forecasted")
    quantiles: Quantiles

class HealthResponse(BaseModel):
    """
    Simple heartbeat response for monitoring tools (e.g. Kubernetes readiness probes).
    """
    status: str = "ok"
    current_time: datetime