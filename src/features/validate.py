import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def validate_feature_set(df: pd.DataFrame, config: dict) -> bool:
    """
    Validates the integrity of the feature set before training.

    Checks performed:
    1. No missing values (NaNs) are present.
    2. Timestamps are strictly monotonic and unique.
    3. No target leakage is detected in lag features.
    
    Args:
        df: The dataframe containing features and target.
        config: The configuration dictionary containing target definitions.
        
    Returns:
        bool: True if all checks pass. Raises ValueError otherwise.
    """
    logger.info("Running feature validation checks...")
    
    target_col = config['target']['name']
    
    # 1. Check for missing values (NaNs)
    # The dataset should be fully cleaned and imputed by this stage.
    if df.isna().any().any():
        nan_counts = df.isna().sum()
        nan_cols = nan_counts[nan_counts > 0]
        msg = f"Validation Failed: NaNs detected in columns:\n{nan_cols}"
        logger.error(msg)
        raise ValueError(msg)

    # 2. Check Timestamp Monotonicity and Uniqueness
    # Time series models require strictly ordered time steps.
    if not df['timestamp'].is_monotonic_increasing:
        msg = "Validation Failed: Timestamps are not strictly increasing."
        logger.error(msg)
        raise ValueError(msg)
    
    if df['timestamp'].duplicated().any():
        dupes = df['timestamp'][df['timestamp'].duplicated()].unique()
        msg = f"Validation Failed: Duplicate timestamps detected: {dupes}"
        logger.error(msg)
        raise ValueError(msg)

    # 3. Target Leakage Check
    # Verify that lag_1 is not identical to the target variable.
    # If they are identical, it implies future information was leaked into the feature.
    lag_1_col = f"{target_col}_lag_1"
    
    if lag_1_col in df.columns:
        # Use numpy.allclose to handle floating point comparisons.
        # We skip the check if the column is missing (e.g., if lag features were disabled).
        
        # Note: We assume the time series is not constant. If the signal is perfectly constant,
        # this check would flag a false positive, but that is rare in carbon intensity data.
        is_leakage = np.allclose(
            df[target_col], 
            df[lag_1_col],
            atol=1e-5,
            equal_nan=True
        )
        
        if is_leakage:
            msg = f"Leakage Detected: {lag_1_col} is identical to {target_col}. Ensure shift(1) was applied."
            logger.error(msg)
            raise ValueError(msg)
            
    logger.info("Validation checks passed.")
    return True

if __name__ == "__main__":
    # Test harness for local debugging
    logging.basicConfig(level=logging.INFO)
    
    # Case 1: Valid Data
    df_good = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=5, freq='h'),
        'carbon': [10.0, 12.0, 15.0, 14.0, 13.0],
        'carbon_lag_1': [9.0, 10.0, 12.0, 15.0, 14.0] 
    })
    
    # Case 2: Leaky Data (Target == Lag_1)
    df_bad = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=5, freq='h'),
        'carbon': [10.0, 12.0, 15.0, 14.0, 13.0],
        'carbon_lag_1': [10.0, 12.0, 15.0, 14.0, 13.0]
    })
    
    cfg = {'target': {'name': 'carbon'}}

    print("--- Testing Valid Data ---")
    try:
        validate_feature_set(df_good, cfg)
    except ValueError as e:
        print(e)
    
    print("\n--- Testing Leaky Data ---")
    try:
        validate_feature_set(df_bad, cfg)
    except ValueError as e:
        print(f"Caught expected error: {e}")