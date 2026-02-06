import requests
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Open-Meteo Archive API (for historical data)
BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
MAX_RETRIES = 3
RETRY_DELAY_SEC = 5 
OUTPUT_DIR = Path("data/raw/weather")

# "Center of GB" coordinates (Dunsop Bridge)
LATITUDE = 54.0
LONGITUDE = -2.5

def fetch_weather_chunk(start_date: str, end_date: str) -> Optional[Dict[str, Any]]:
    """
    Fetches hourly weather variables relevant to energy generation:
    - temperature_2m: affects demand (heating/cooling)
    - wind_speed_10m: affects wind generation
    - direct_radiation: affects solar generation
    """
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["temperature_2m", "wind_speed_10m", "cloudcover"],
        "timezone": "GMT"
    }

    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"Requesting weather: {start_date} to {end_date} (Attempt {attempt + 1})")
            response = requests.get(BASE_URL, params=params, timeout=15)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                logger.warning("Rate limit hit (429). Cooling down for 60s...")
                time.sleep(60)
            else:
                logger.warning(f"API Error {response.status_code}: {response.text}")
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error: {e}")
            
        time.sleep(RETRY_DELAY_SEC * (attempt + 1))

    logger.error(f"Failed to fetch weather for {start_date}")
    return None

def save_data(data: Dict[str, Any], filename: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    file_path = OUTPUT_DIR / filename
    
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved {file_path}")

def run_ingestion(start_date: str, end_date: str):
    """
    Open-Meteo can handle large ranges, but we chunk by Month
    to keep file sizes manageable and consistent with MLOps versioning.
    """
    try:
        current = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        logger.error("Invalid date format. Use YYYY-MM-DD")
        return

    while current <= end:
        # Calculate chunk end (end of current month or end_date)
        next_month = (current.replace(day=1) + timedelta(days=32)).replace(day=1)
        chunk_end = min(end, next_month - timedelta(days=1))
        
        s_str = current.strftime("%Y-%m-%d")
        e_str = chunk_end.strftime("%Y-%m-%d")
        
        data = fetch_weather_chunk(s_str, e_str)
        
        if data:
            save_data(data, f"weather_{s_str}_{e_str}.json")
        
        # Move to first day of next month
        current = next_month
        
        # Respect Open-Meteo 'Fair Use' (avoid hammering)
        time.sleep(2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Historical Weather (Open-Meteo)")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    
    args = parser.parse_args()
    run_ingestion(args.start, args.end)