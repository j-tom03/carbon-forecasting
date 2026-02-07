import torch
import json
import logging
from pathlib import Path
from functools import lru_cache
from typing import Tuple, Dict, Any

# Import the architecture definition
from models.tft import TemporalFusionTransformer

logger = logging.getLogger("uvicorn")

# Define where the production model lives (relative to this file)
# src/api/dependencies.py -> src/api/ -> src/ -> PROJECT_ROOT
PROJECT_ROOT = Path(__file__).parent.parent.parent
PROD_MODEL_DIR = PROJECT_ROOT / "models" / "tft_prod"

@lru_cache()
def get_model_artifacts() -> Tuple[TemporalFusionTransformer, Dict[str, Any]]:
    """
    Loads the production model and metadata.
    
    This function is cached, so it acts as a Singleton. 
    The first request triggers the load (cold start); subsequent requests use the cache.
    
    Returns:
        model: The loaded PyTorch model (eval mode, on CPU).
        meta: The metadata dictionary (version, config, scaling info).
        
    Raises:
        FileNotFoundError: If the production artifacts are missing.
    """
    logger.info(f"Loading model artifacts from {PROD_MODEL_DIR}...")

    # 1. Validation: Fail fast if files are missing
    meta_path = PROD_MODEL_DIR / "metadata.json"
    weights_path = PROD_MODEL_DIR / "model.pt"

    if not meta_path.exists() or not weights_path.exists():
        error_msg = (
            f"Critical Error: Production artifacts not found at {PROD_MODEL_DIR}. "
            "Did you run 'scripts/run_hpo.py' or 'scripts/train.py'?"
        )
        logger.critical(error_msg)
        raise FileNotFoundError(error_msg)

    # 2. Load Metadata (The "Instruction Manual")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    # 3. Initialise Model Architecture
    # The __init__ expects a config dict with 'model' and 'loss' keys.
    # Our metadata stores the 'model' config directly.
    model_config = meta['config']
    data_spec = meta['data_spec']
    
    # Reconstruct the exact structure the class expects
    full_config = {
        'model': model_config,
        'loss': {'quantiles': [0.1, 0.5, 0.9]} # Default quantiles
    }

    model = TemporalFusionTransformer(
        config=full_config,
        num_features=data_spec['num_features'],
        output_horizon=data_spec['horizon']
    )

    # 4. Load Weights
    # map_location='cpu' for API servers
    try:
        state_dict = torch.load(weights_path, map_location=torch.device('cpu'), weights_only=True)
        model.load_state_dict(state_dict)
        model.eval() # Freezes layers like Dropout (crucial for consistent inference)
        
        version = meta.get('version', 'unknown')
        logger.info(f"Model loaded successfully (Version: {version})")
        
    except Exception as e:
        logger.error(f"Failed to load model state: {e}")
        raise RuntimeError("Model weight loading failed") from e

    return model, meta