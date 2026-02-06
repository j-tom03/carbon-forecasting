import pandas as pd
import numpy as np
import torch
import logging
import yaml
import sys
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict

# Add src to path for internal imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(PROJECT_ROOT / "src"))

from features import time_features, lag_features, rolling_features, validate

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [TRAINING] - %(message)s")
logger = logging.getLogger(__name__)

class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, lookback: int, horizon: int):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.lookback = lookback
        self.horizon = horizon

    def __len__(self):
        return len(self.X) - self.lookback - self.horizon + 1

    def __getitem__(self, idx):
        x_start = idx
        x_end = idx + self.lookback
        y_start = x_end
        y_end = y_start + self.horizon
        return self.X[x_start:x_end], self.y[y_start:y_end]

def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def build_dataset(config_path: Path):
    """
    Main entry point for building the dataset.
    """
    logger.info(f"Starting Dataset Build using config: {config_path}")
    
    # 1. Load Config
    config = load_config(config_path)
    
    # Define paths based on project root, not hardcoded
    # We assume standard project structure: data/processed is sibling to src/
    processed_data_path = PROJECT_ROOT / "data" / "processed" / "dataset_v1.parquet"
    output_dir = PROJECT_ROOT / "data" / "training"
    
    if not processed_data_path.exists():
        logger.error(f"Input file not found: {processed_data_path}")
        sys.exit(1)
        
    df = pd.read_parquet(processed_data_path)
    logger.info(f"Loaded processed data: {len(df)} rows")

    # 2. Feature Engineering
    logger.info("Applying feature transformations...")
    df = time_features.add_time_features(df, time_col="timestamp")
    
    target_col = config['target']['name']
    
    # Lags
    if 'lag_features' in config and target_col in config['lag_features']:
        lags = config['lag_features'][target_col]
        df = lag_features.add_lag_features(df, target_col, lags)
    
    # Rolling
    if 'rolling_features' in config and target_col in config['rolling_features']:
        rolling_specs = config['rolling_features'][target_col]
        df = rolling_features.add_rolling_features(df, target_col, rolling_specs)
        
    # Drop Warmup
    initial_len = len(df)
    df = df.dropna()
    logger.info(f"Dropped {initial_len - len(df)} rows due to warmup.")

    # 3. Validation
    try:
        validate.validate_feature_set(df, config)
    except ValueError as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)

    # 4. Tensor Preparation
    lookback = config['lookback_steps']
    horizon = config['forecast_horizon']
    
    df = df.sort_values("timestamp")
    feature_cols = [c for c in df.columns if c != "timestamp"]
    
    X_raw = df[feature_cols].values
    y_raw = df[target_col].values.reshape(-1, 1)

    # 5. Split
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    # 6. Scaling
    logger.info("Fitting Scalers...")
    scaler_X = StandardScaler()
    scaler_X.fit(X_raw[:train_end])
    X_scaled = scaler_X.transform(X_raw)
    
    scaler_y = StandardScaler()
    scaler_y.fit(y_raw[:train_end])
    y_scaled = scaler_y.transform(y_raw).flatten()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler_X, output_dir / "scaler_X.save")
    joblib.dump(scaler_y, output_dir / "scaler_y.save")

    # 7. Save Datasets
    splits = {
        "train": (X_scaled[:train_end], y_scaled[:train_end]),
        "val":   (X_scaled[train_end:val_end], y_scaled[train_end:val_end]),
        "test":  (X_scaled[val_end:], y_scaled[val_end:])
    }
    
    for split_name, (X_split, y_split) in splits.items():
        if len(X_split) < (lookback + horizon):
            continue
        dataset = TimeSeriesDataset(X_split, y_split, lookback, horizon)
        torch.save(dataset, output_dir / f"{split_name}.pt")
        logger.info(f"Saved {split_name}.pt ({len(dataset)} sequences)")

    # 8. Metadata
    meta = {
        "num_features": len(feature_cols),
        "feature_names": feature_cols,
        "lookback": lookback,
        "horizon": horizon,
        "train_samples": len(splits['train'][0]),
        "scaling": "standard_scaler"
    }
    with open(output_dir / "meta.yaml", "w") as f:
        yaml.dump(meta, f)
        
    logger.info("Dataset build complete.")