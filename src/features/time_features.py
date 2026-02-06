import pandas as pd
import logging

logger = logging.getLogger(__name__)

def add_time_features(df: pd.DataFrame, time_col: str = "timestamp") -> pd.DataFrame:
    """
    Adds deterministic time-based features to the dataframe.
    
    Args:
        df: Input dataframe containing the timestamp column.
        time_col: Name of the column containing datetime objects.
        
    Returns:
        pd.DataFrame: Dataframe with added time features.
    """
    # Ensure we work on a copy to avoid SettingWithCopy warnings on slices
    df = df.copy()

    if time_col not in df.columns:
        raise ValueError(f"Column '{time_col}' not found in dataframe.")

    # Ensure correct type
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])

    logger.info("Generating time-based features...")

    # 1. Hour of Day (0-23)
    # Critical for capturing daily cycles (morning peak vs night trough)
    df['hour_of_day'] = df[time_col].dt.hour.astype('int16')

    # 2. Day of Week (0=Monday, 6=Sunday)
    # Critical for capturing weekly cycles (weekday vs weekend)
    df['day_of_week'] = df[time_col].dt.dayofweek.astype('int16')

    # 3. Weekend Flag (Boolean/Int)
    # Useful because demand drops significantly on Sat/Sun
    df['is_weekend'] = (df['day_of_week'] >= 5).astype('int8')

    # 4. Month (1-12) - (Optional but often useful for seasonality)
    df['month'] = df[time_col].dt.month.astype('int16')

    logger.info(f"Added features: hour_of_day, day_of_week, is_weekend, month.")
    
    return df

if __name__ == "__main__":
    # Quick test harness
    logging.basicConfig(level=logging.INFO)
    
    # Create dummy data
    dates = pd.date_range(start="2024-01-01", periods=5, freq="h")
    test_df = pd.DataFrame({"timestamp": dates, "value": range(5)})
    
    processed = add_time_features(test_df)
    print(processed.head())