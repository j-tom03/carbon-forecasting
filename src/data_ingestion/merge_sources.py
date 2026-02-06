import pandas as pd
import logging
import json
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
OUTPUT_PARQUET = PROCESSED_DIR / "dataset_v1.parquet"
OUTPUT_META = PROCESSED_DIR / "dataset_v1_meta.json"

def merge_sources():
    logger.info("Starting merge process...")
    
    # 1. Load Silver Tables
    try:
        carbon_df = pd.read_parquet(PROCESSED_DIR / "carbon.parquet")
        weather_df = pd.read_parquet(PROCESSED_DIR / "weather.parquet")
    except FileNotFoundError as e:
        logger.error(f"Missing input files. Run normalise.py first. Error: {e}")
        return
    
    # deduplication of timestamps
    before_count = len(carbon_df)
    carbon_df = carbon_df.drop_duplicates(subset=['timestamp'], keep='last')
    deduped_count = len(carbon_df)
    
    if before_count != deduped_count:
        logger.warning(f"Removed {before_count - deduped_count} duplicate timestamps from Carbon data.")

    # 2. Prepare Weather (Resample & Forward Fill)
    weather_df = weather_df.set_index('timestamp')
    weather_resampled = weather_df.resample('30min').ffill()
    
    # 3. Merge (Left Join on Carbon)
    merged = pd.merge(
        carbon_df, 
        weather_resampled, 
        left_on='timestamp', 
        right_index=True, 
        how='left'
    )
    
    # 4. Add Quality Flags
    merged['weather_missing'] = merged['temperature'].isna().astype(int)
    merged['carbon_missing'] = merged['carbon_intensity'].isna().astype(int)

    # 5. Save Main Dataset (Parquet)
    merged.to_parquet(OUTPUT_PARQUET, index=False)
    logger.info(f"Saved Dataset: {OUTPUT_PARQUET}")

    # 6. Generate & Save Metadata (The "Sidecar")
    # This creates the audit trail required for MLOps
    metadata = {
        "creation_timestamp": datetime.utcnow().isoformat(),
        "start_date": merged['timestamp'].min().isoformat(),
        "end_date": merged['timestamp'].max().isoformat(),
        "resolution": "30min",
        "rows": len(merged),
        "columns": merged.columns.tolist(),
        "sources": [
            "National Grid ESO (Carbon)",
            "Open-Meteo (Weather)"
        ],
        "quality_report": {
            "missing_weather_rows": int(merged['weather_missing'].sum()),
            "missing_carbon_rows": int(merged['carbon_missing'].sum())
        }
    }

    with open(OUTPUT_META, "w") as f:
        json.dump(metadata, f, indent=4)
        
    logger.info(f"Saved Metadata Sidecar: {OUTPUT_META}")

if __name__ == "__main__":
    merge_sources()