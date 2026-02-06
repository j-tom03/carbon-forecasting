import pandas as pd
import logging
from typing import List

logger = logging.getLogger(__name__)

def add_lag_features(df: pd.DataFrame, target_col: str, lag_steps: List[int]) -> pd.DataFrame:
    """
    Adds past values (lags) of the target variable as features.
    
    Args:
        df: Input dataframe.
        target_col: The column to create lags for (e.g., 'carbon_intensity').
        lag_steps: List of integers representing time steps to look back (e.g. [1, 48]).
        
    Returns:
        pd.DataFrame: Dataframe with new columns like 'carbon_intensity_lag_1'.
    """
    # Work on a copy to prevent side effects
    df = df.copy()
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")
    
    logger.info(f"Generating {len(lag_steps)} lag features for '{target_col}'...")
    
    for lag in lag_steps:
        # Naming convention: {variable}_lag_{steps}
        feature_name = f"{target_col}_lag_{lag}"
        
        # shift(k): takes value at t-k and moves it to t
        # This ensures we are strictly using PAST data
        df[feature_name] = df[target_col].shift(lag)
        
    return df

if __name__ == "__main__":
    # Test harness
    logging.basicConfig(level=logging.INFO)
    
    # Dummy data: 0, 10, 20, 30, 40
    test_df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=5, freq="h"),
        "carbon_intensity": [0, 10, 20, 30, 40]
    })
    
    # Test lags of 1 hour and 2 hours
    processed = add_lag_features(test_df, "carbon_intensity", lag_steps=[1, 2])
    print(processed)