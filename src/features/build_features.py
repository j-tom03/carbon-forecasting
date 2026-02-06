import pandas as pd
import yaml
import logging
import sys
from pathlib import Path

# Add the src directory to the python path to allow imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(PROJECT_ROOT / "src"))

from features import time_features, lag_features, rolling_features, validate

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [FEATURES] - %(message)s")
logger = logging.getLogger(__name__)

# Define file paths
CONFIG_PATH = PROJECT_ROOT / "configs" / "dataset_v1.yaml"
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "dataset_v1.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "features"
OUTPUT_PATH = OUTPUT_DIR / "features_v1.parquet"

def load_config(path: Path) -> dict:
    """Load the YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)

def build_features():
    logger.info("Starting Feature Engineering Pipeline...")
    
    # Load the configuration and the processed dataset
    if not INPUT_PATH.exists():
        logger.error(f"Input file not found: {INPUT_PATH}")
        sys.exit(1)
        
    df = pd.read_parquet(INPUT_PATH)
    config = load_config(CONFIG_PATH)
    logger.info(f"Loaded input data: {len(df)} rows")

    # Add time-based features (hour, day, etc.)
    # These are derived directly from the timestamp
    df = time_features.add_time_features(df, time_col="timestamp")

    # Add lag features based on the configuration
    target_col = config['target']['name']
    if 'lag_features' in config and target_col in config['lag_features']:
        lags = config['lag_features'][target_col]
        df = lag_features.add_lag_features(df, target_col, lags)
    
    # Add rolling window features based on the configuration
    if 'rolling_features' in config and target_col in config['rolling_features']:
        rolling_specs = config['rolling_features'][target_col]
        df = rolling_features.add_rolling_features(df, target_col, rolling_specs)

    # Drop the initial rows that contain missing values due to lags
    # This ensures the model receives a complete dataset without NaNs
    initial_len = len(df)
    df = df.dropna()
    dropped_rows = initial_len - len(df)
    logger.info(f"Dropped {dropped_rows} rows due to lag/rolling warmup period.")

    # Validate the feature set to ensure no data leakage or errors
    try:
        validate.validate_feature_set(df, config)
    except ValueError as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)

    # Save the final feature set to a parquet file
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    logger.info(f"Saved Feature Set: {OUTPUT_PATH}")
    logger.info(f"Final Shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")

if __name__ == "__main__":
    build_features()