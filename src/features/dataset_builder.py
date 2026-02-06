import pandas as pd
import numpy as np
import torch
import logging
import yaml
import sys
import joblib  # Added for saving the scaler
from pathlib import Path
from sklearn.preprocessing import StandardScaler # Added for scaling

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(PROJECT_ROOT / "src"))

# Import your feature modules
from features import time_features, lag_features, rolling_features, validate

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [TRAINING] - %(message)s")
logger = logging.getLogger(__name__)

# Paths
CONFIG_PATH = PROJECT_ROOT / "configs" / "dataset_v1.yaml"
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "dataset_v1.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "training"

def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

class TimeSeriesDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset that creates sliding windows over the time series.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, lookback: int, horizon: int):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.lookback = lookback
        self.horizon = horizon

    def __len__(self):
        return len(self.X) - self.lookback - self.horizon + 1

    def __getitem__(self, idx):
        # Input: [t-lookback : t]
        x_start = idx
        x_end = idx + self.lookback
        
        # Target: [t : t+horizon]
        y_start = x_end
        y_end = y_start + self.horizon
        
        return self.X[x_start:x_end], self.y[y_start:y_end]

def build_sequences():
    logger.info("Starting Sequence Builder...")
    
    # 1. Load Config & Data
    config = load_config(CONFIG_PATH)
    if not INPUT_PATH.exists():
        logger.error(f"Input file not found: {INPUT_PATH}")
        sys.exit(1)
        
    df = pd.read_parquet(INPUT_PATH)
    logger.info(f"Loaded processed data: {len(df)} rows")

    # 2. FEATURE ENGINEERING
    logger.info("Applying feature transformations...")
    df = time_features.add_time_features(df, time_col="timestamp")
    
    target_col = config['target']['name']
    if 'lag_features' in config and target_col in config['lag_features']:
        lags = config['lag_features'][target_col]
        df = lag_features.add_lag_features(df, target_col, lags)
    
    if 'rolling_features' in config and target_col in config['rolling_features']:
        rolling_specs = config['rolling_features'][target_col]
        df = rolling_features.add_rolling_features(df, target_col, rolling_specs)
        
    # Drop Warmup
    initial_len = len(df)
    df = df.dropna()
    logger.info(f"Dropped {initial_len - len(df)} rows due to warmup.")

    # 3. VALIDATION
    try:
        validate.validate_feature_set(df, config)
    except ValueError as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)

    # 4. PREPARE VECTORS
    lookback = config['lookback_steps']
    horizon = config['forecast_horizon']
    
    # Sort strictly by time
    df = df.sort_values("timestamp")
    
    # Separate Features (X) and Target (y)
    # We drop timestamp from X, but keep everything else
    feature_cols = [c for c in df.columns if c != "timestamp"]
    
    X_raw = df[feature_cols].values
    y_raw = df[target_col].values.reshape(-1, 1) # Reshape for scaler

    # 5. TEMPORAL SPLIT INDICES
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    logger.info(f"Splitting: Train ({train_end}), Val ({val_end-train_end}), Test ({n-val_end})")

    # 6. SCALING (The Missing Piece from Step 7)
    # Fit scaler ONLY on training data to prevent leakage
    logger.info("Fitting Scalers...")
    
    # Scale Features
    scaler_X = StandardScaler()
    scaler_X.fit(X_raw[:train_end])
    X_scaled = scaler_X.transform(X_raw)
    
    # Scale Target (Optional but recommended for faster convergence)
    scaler_y = StandardScaler()
    scaler_y.fit(y_raw[:train_end])
    y_scaled = scaler_y.transform(y_raw).flatten() # Flatten back to 1D
    
    # Save Scalers for Inference API
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler_X, OUTPUT_DIR / "scaler_X.save")
    joblib.dump(scaler_y, OUTPUT_DIR / "scaler_y.save")
    logger.info("Saved scaling parameters.")

    # 7. CREATE DATASETS & SAVE
    splits = {
        "train": (X_scaled[:train_end], y_scaled[:train_end]),
        "val":   (X_scaled[train_end:val_end], y_scaled[train_end:val_end]),
        "test":  (X_scaled[val_end:], y_scaled[val_end:])
    }
    
    for split_name, (X_split, y_split) in splits.items():
        if len(X_split) < (lookback + horizon):
            continue
            
        dataset = TimeSeriesDataset(X_split, y_split, lookback, horizon)
        torch.save(dataset, OUTPUT_DIR / f"{split_name}.pt")
        logger.info(f"Saved {split_name}.pt ({len(dataset)} sequences)")

    # 8. SAVE METADATA
    meta = {
        "num_features": len(feature_cols),
        "feature_names": feature_cols,
        "lookback": lookback,
        "horizon": horizon,
        "train_samples": len(splits['train'][0]),
        "scaling": "standard_scaler",
        "date_range": {
            "start": str(df['timestamp'].iloc[0]),
            "end": str(df['timestamp'].iloc[-1])
        }
    }
    with open(OUTPUT_DIR / "meta.yaml", "w") as f:
        yaml.dump(meta, f)
        
    logger.info("Dataset build complete.")

if __name__ == "__main__":
    build_sequences()