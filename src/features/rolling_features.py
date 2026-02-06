import pandas as pd
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

def add_rolling_features(df: pd.DataFrame, target_col: str, window_specs: List[Dict]) -> pd.DataFrame:
    """
    Adds rolling window statistics (mean, std) to the dataframe.
    
    Args:
        df: Input dataframe.
        target_col: The column to compute statistics on.
        window_specs: List of dicts defining windows. 
                      Example: [{'window': 48, 'stats': ['mean', 'std']}]
                      
    Returns:
        pd.DataFrame: Dataframe with new columns like 'carbon_rolling_mean_48'.
    """
    # Work on a copy to avoid SettingWithCopy warnings
    df = df.copy()
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
        
    logger.info(f"Generating rolling features for '{target_col}'...")

    for spec in window_specs:
        window_size = spec['window']
        stats = spec['stats']
        
        # 1. Shift data
        # We move t-1 to position t.
        # This ensures the rolling window looking backwards from t ONLY sees past data.
        shifted_series = df[target_col].shift(1)
        
        # 2. Define the Roller
        # min_periods ensures we don't return NaN just because 1 hour is missing in a 24h window
        roller = shifted_series.rolling(window=window_size, min_periods=window_size // 2)
        
        for stat in stats:
            feature_name = f"{target_col}_rolling_{stat}_{window_size}"
            
            if stat == 'mean':
                df[feature_name] = roller.mean()
            elif stat == 'std':
                df[feature_name] = roller.std()
            elif stat == 'min':
                df[feature_name] = roller.min()
            elif stat == 'max':
                df[feature_name] = roller.max()
            
            # 3. Memory Optimisation
            # Rolling ops often output float64; reducing to float32 saves 50% RAM
            df[feature_name] = df[feature_name].astype('float32')

    return df

if __name__ == "__main__":
    # Test Harness to verify NO LEAKAGE
    logging.basicConfig(level=logging.INFO)
    
    # Create a simple sequence: 0, 10, 20, 30, 40
    data = {
        "timestamp": pd.date_range("2024-01-01", periods=5, freq="h"),
        "carbon": [0, 10, 20, 30, 40]
    }
    test_df = pd.DataFrame(data)
    
    print("Original Data:")
    print(test_df)
    
    # Calculate rolling mean of last 2 hours
    # At index 2 (value 20), we expect mean(0, 10) = 5.0
    # If we leaked the current value, we would get mean(10, 20) = 15.0
    
    specs = [{'window': 2, 'stats': ['mean']}]
    processed = add_rolling_features(test_df, "carbon", specs)
    
    print("\nProcessed Data (Watch row 2):")
    print(processed[['timestamp', 'carbon', 'carbon_rolling_mean_2']])