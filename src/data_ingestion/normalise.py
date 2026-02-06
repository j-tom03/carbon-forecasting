import pandas as pd
import json
import logging
import glob
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths matching your structure
RAW_CARBON_DIR = Path("data/raw/carbon_intensity")
RAW_WEATHER_DIR = Path("data/raw/weather")
PROCESSED_DIR = Path("data/processed")

def normalize_carbon():
    """Reads raw Carbon JSONs and outputs a clean Parquet file."""
    logger.info("Normalizing Carbon Data...")
    
    files = sorted(glob.glob(str(RAW_CARBON_DIR / "*.json")))
    records = []
    
    for f in files:
        try:
            with open(f, 'r') as file:
                data = json.load(file)
                # Extract only what we need: timestamp and actual intensity
                if 'data' in data:
                    for entry in data['data']:
                        records.append({
                            'timestamp': entry['from'],
                            'carbon_intensity': entry['intensity']['actual']
                        })
        except Exception as e:
            logger.warning(f"Skipping corrupt file {f}: {e}")
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    # 1. Standardize Timestamp (UTC)
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_convert(None)
    
    # 2. Type Enforcement
    df['carbon_intensity'] = pd.to_numeric(df['carbon_intensity'], errors='coerce')
    
    # 3. Clean-up
    df = df.dropna(subset=['carbon_intensity'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Save
    output_path = PROCESSED_DIR / "carbon.parquet"
    df.to_parquet(output_path, index=False)
    logger.info(f"✅ Saved Carbon Silver Table: {output_path} ({len(df)} rows)")

def normalize_weather():
    """Reads raw Weather JSONs and outputs a clean Parquet file."""
    logger.info("Normalizing Weather Data...")
    
    files = sorted(glob.glob(str(RAW_WEATHER_DIR / "*.json")))
    all_dfs = []
    
    for f in files:
        try:
            with open(f, 'r') as file:
                data = json.load(file)
                if 'hourly' in data:
                    # Create a mini dataframe for this chunk
                    chunk = pd.DataFrame({
                        'timestamp': data['hourly']['time'],
                        'temperature': data['hourly']['temperature_2m'],
                        'windspeed': data['hourly']['wind_speed_10m'],
                        'cloudcover': data['hourly']['cloudcover']
                    })
                    all_dfs.append(chunk)
        except KeyError as e:
             logger.warning(f"Missing column in {f}. Did you update fetch_weather.py? Error: {e}")
        except Exception as e:
             logger.warning(f"Skipping corrupt file {f}: {e}")

    if not all_dfs:
        logger.error("No weather data found!")
        return

    # Combine all chunks
    df = pd.concat(all_dfs, ignore_index=True)
    
    # 1. Standardize Timestamp (UTC)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 2. Type Enforcement (Float32 saves memory)
    df['temperature'] = df['temperature'].astype('float32')
    df['windspeed'] = df['windspeed'].astype('float32')
    df['cloudcover'] = df['cloudcover'].astype('float32')
    
    # 3. Clean-up
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Save
    output_path = PROCESSED_DIR / "weather.parquet"
    df.to_parquet(output_path, index=False)
    logger.info(f"✅ Saved Weather Silver Table: {output_path} ({len(df)} rows)")

if __name__ == "__main__":
    # Ensure output directory exists
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    normalize_carbon()
    normalize_weather()