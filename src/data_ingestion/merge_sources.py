import pandas as pd
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
OUTPUT_FILE = PROCESSED_DIR / "merged_dataset.parquet"

def merge_sources():
    logger.info("Starting merge process...")
    
    # 1. Load the "Silver" Parquet files
    try:
        carbon_df = pd.read_parquet(PROCESSED_DIR / "carbon.parquet")
        weather_df = pd.read_parquet(PROCESSED_DIR / "weather.parquet")
        logger.info(f"Loaded: Carbon ({len(carbon_df)} rows), Weather ({len(weather_df)} rows)")
    except FileNotFoundError as e:
        logger.error(f"Missing input files. Run normalise.py first. Error: {e}")
        return

    # 2. Prepare Weather for Merge (Resample & Forward Fill)
    # Weather is Hourly, Carbon is 30-min. We need to match Carbon's frequency.
    
    # Set index to timestamp to enable resampling
    weather_df = weather_df.set_index('timestamp')
    
    # Resample to 30 mins and forward-fill (as per requirements)
    # logical logic: 10:00 weather applies to 10:00 and 10:30
    weather_resampled = weather_df.resample('30min').ffill()
    
    logger.info("Resampled weather from 1h -> 30min (Forward Fill applied).")

    # 3. Perform the Merge (Left Join on Carbon)
    # We maintain the carbon timeline exactly as is.
    merged = pd.merge(
        carbon_df, 
        weather_resampled, 
        left_on='timestamp', 
        right_index=True, 
        how='left'
    )
    
    # 4. Data Quality Flags (The "MLOps Signal")
    # Instead of dropping rows, we flag them.
    
    # Check for missing weather (e.g. if carbon data exists but weather file didn't cover that date)
    merged['weather_missing'] = merged['temperature'].isna().astype(int)
    
    # Check for missing carbon (rare, but if the raw file had NaNs that weren't dropped in normalisation)
    merged['carbon_missing'] = merged['carbon_intensity'].isna().astype(int)

    # 5. Logging Data Quality
    n_weather_missing = merged['weather_missing'].sum()
    n_carbon_missing = merged['carbon_missing'].sum()
    
    if n_weather_missing > 0:
        logger.warning(f"{n_weather_missing} rows are missing weather data. Flags added.")
    else:
        logger.info("All carbon timestamps have matching weather data.")

    if n_carbon_missing > 0:
        logger.warning(f"{n_carbon_missing} rows have missing carbon intensity. Flags added.")

    # 6. Save "Gold" Dataset
    merged.to_parquet(OUTPUT_FILE, index=False)
    logger.info(f"Saved merged dataset to {OUTPUT_FILE}")
    logger.info(f"Final Shape: {merged.shape}")
    logger.info(f"Columns: {merged.columns.tolist()}")

if __name__ == "__main__":
    merge_sources()