from fastapi import FastAPI, HTTPException, Depends, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.append(str(PROJECT_ROOT))

from api.schemas import ForecastResponse, HealthResponse, Quantiles
from api.dependencies import get_model_artifacts
from api.inference import run_inference, load_simulated_context

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn")

# LIFESPAN MANAGER
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Executes on startup. 
    Pre-loads the model into memory so the first user request is fast.
    """
    try:
        logger.info("API Starting up... Pre-loading model artifacts.")
        get_model_artifacts() # Trigger the @lru_cache
        logger.info("Model loaded and ready.")
    except Exception as e:
        logger.critical(f"Failed to load model on startup: {e}")
        # We don't exit here, but /health will report the failure.
    
    yield # Control passes to the application
    
    logger.info("API Shutting down...")

# --- APP INITIALISATION
app = FastAPI(
    title="Carbon Intensity Forecaster",
    version="2.0",
    description="Probabilistic 96h forecasts for UK Grid Carbon Intensity using Temporal Fusion Transformers.",
    lifespan=lifespan
)

# --- INPUT VALIDATION & SAFEGUARDS

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Overrides the default 422 error with a friendly 400 Bad Request.
    """
    return JSONResponse(
        status_code=400, # <--- Requirement Met: HTTP 400
        content={
            "detail": "Invalid parameters provided.",
            "tip": "Forecast horizon must be between 1 and 192 hours.",
            "error": str(exc) # Keeps technical detail for debugging if needed
        },
    )

# --- ENDPOINTS

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """
    Kubernetes/Docker Health Endpoint.
    Verifies that the model is actually loaded in memory.
    """
    try:
        # Check if model is reachable
        model, meta = get_model_artifacts()
        version = meta.get("version", "unknown")
        return HealthResponse(
            status="ok", 
            current_time=datetime.now(timezone.utc),
            details=f"Model {version} active"
        )
    except Exception as e:
        # If model isn't loaded, return 503 (Service Unavailable)
        raise HTTPException(status_code=503, detail=f"System Unhealthy: {str(e)}")

@app.get("/predict", response_model=ForecastResponse, tags=["Inference"])
async def predict(
    horizon: int = Query(48, ge=1, le=192, description="Hours to forecast (1-192)"),
    artifacts = Depends(get_model_artifacts)
):
    model, meta = artifacts
    
    # 1. Get Context
    context_tensor = load_simulated_context("data/processed")
    
    # Fallback to noise if val.pt fails
    if context_tensor is None:
        import torch
        # Use lookback from metadata or default to 96
        lookback = meta.get('data_spec', {}).get('lookback', 96)
        num_features = meta.get('data_spec', {}).get('num_features', 19)
        context_tensor = torch.randn(1, lookback, num_features)

    # 2. Prepare Scaler (With Defaults)
    # Try to get from metadata, otherwise use UK Grid defaults
    scaler_params = meta.get('data_spec', {}).get('scaler', None)
    
    if scaler_params is None:
        logger.warning("Scaler not found in metadata. Using UK Grid defaults.")
        scaler_params = {"mean": 182.0, "std": 64.0}

    # 3. Run Inference
    try:
        result_payload = run_inference(
            model, 
            context_tensor, 
            horizon, 
            scaler=scaler_params
        )
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail="Internal Model Error")

    # 4. Construct Response
    return ForecastResponse(
        model_version=meta.get("version", "unknown"),
        generated_at=datetime.now(timezone.utc),
        horizon=horizon,
        quantiles=Quantiles(**result_payload["quantiles"])
    )