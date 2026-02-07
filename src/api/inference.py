import torch
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any

from models.tft import TemporalFusionTransformer

logger = logging.getLogger("uvicorn")

def run_inference(
    model: TemporalFusionTransformer, 
    context_tensor: torch.Tensor, 
    horizon: int,
    scaler: Dict[str, float] = None) -> Dict[str, Any]:
    
    # 1. Validation
    max_horizon = model.output_horizon
    if horizon > max_horizon: horizon = max_horizon

    # 2. Forward Pass
    with torch.no_grad():
        prediction = model(context_tensor)
    
    # 3. Unscale
    raw_output = prediction[0].numpy()
    
    if scaler:
        mean = scaler.get('mean', 0.0)
        std = scaler.get('std', 1.0)
        raw_output = (raw_output * std) + mean
        raw_output = np.maximum(raw_output, 0.0)

    sliced_output = raw_output[:horizon, :]
    
    return {
        "quantiles": {
            "p10": sliced_output[:, 0].tolist(),
            "p50": sliced_output[:, 1].tolist(),
            "p90": sliced_output[:, 2].tolist(),
        }
    }

def load_simulated_context(data_dir_path: str) -> torch.Tensor:
    try:
        # Robust path finding
        project_root = Path(data_dir_path).parent.parent 
        val_path = project_root / "data" / "training" / "val.pt"
        
        if not val_path.exists():
            val_path = Path("data/training/val.pt")

        if not val_path.exists():
            logger.warning(f"val.pt not found at {val_path}")
            return None

        logger.info(f"Loading real context from {val_path}")
        
        # We allow loading custom classes (like TimeSeriesDataset)
        data = torch.load(val_path, map_location='cpu', weights_only=False)
        
        X = None
        
        # Case A: It is a Dataset object (has __getitem__)
        if hasattr(data, "__getitem__") and hasattr(data, "__len__"):
            # Pick a random sample index
            idx = np.random.randint(0, len(data))
            
            # Datasets usually return (X, y) tuple
            sample = data[idx]
            if isinstance(sample, (tuple, list)):
                X = sample[0] # Take the input tensor
            else:
                X = sample
        
        # Case B: It is a raw TensorDataset wrapper
        elif hasattr(data, 'tensors'):
            X = data.tensors[0]
            idx = np.random.randint(0, len(X))
            X = X[idx]

        # Case C: It is just a raw Tensor
        elif isinstance(data, torch.Tensor):
            X = data
            idx = np.random.randint(0, len(X))
            X = X[idx]

        else:
            logger.error(f"Unknown data format: {type(data)}")
            return None

        # Ensure we have the batch dimension [1, Lookback, Features]
        if X.dim() == 2:
            X = X.unsqueeze(0)
            
        return X

    except Exception as e:
        logger.error(f"Failed to load val.pt: {e}")
        return None