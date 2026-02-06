import requests
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any
import argparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

BASE_URL = "https://api.carbonintensity.org.uk"
MAX_RETRIES = 3
RETRY_DELAY_SEC = 2
OUTPUT_DIR = Path("data/raw/carbon_intensity")

def fetch_intensity_data(start_iso: str, end_iso: str) -> Optional[Dict[str, Any]]:
    """
    Fetches carbon intensity data for a specific time window.
    """
    url = f"{BASE_URL}/intensity/{start_iso}/{end_iso}"
    
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"Requesting data: {url} (Attempt {attempt + 1}/{MAX_RETRIES})")
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 400:
                logger.error(f"Bad Request (400): {response.text}")
                return None
            else:
                logger.warning(f"API Error {response.status_code}: {response.text}. Retrying...")
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error: {e}. Retrying...")
            
        time.sleep(RETRY_DELAY_SEC * (attempt + 1))

    logger.error(f"Failed to fetch data for range {start_iso} to {end_iso} after {MAX_RETRIES} attempts.")
    return None

def save_raw_data(data: Dict[str, Any], date_str: str) -> None:
    """
    Saves raw API response to JSON.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    file_path = OUTPUT_DIR / f"{date_str}.json"
    
    try:
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Successfully saved data to {file_path}")
    except IOError as e:
        logger.error(f"Failed to write file {file_path}: {e}")

def run_ingestion(start_date: str, end_date: str):
    """
    Iterates through dates day-by-day to respect chunking and file naming conventions.
    """
    try:
        current_date = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        logger.error("Invalid date format. Please use YYYY-MM-DD.")
        return

    if current_date > end:
        logger.error("Start date cannot be after end date.")
        return

    logger.info(f"Starting ingestion from {start_date} to {end_date}")

    while current_date <= end:
        # Define window for a single day (Start of day to End of day)
        # Using ISO8601 format required by API
        window_start = current_date.strftime("%Y-%m-%dT00:00Z")
        window_end = current_date.strftime("%Y-%m-%dT23:59Z")
        date_str = current_date.strftime("%Y-%m-%d")

        data = fetch_intensity_data(window_start, window_end)
        
        if data and 'data' in data:
            save_raw_data(data, date_str)
        else:
            logger.warning(f"No valid data returned for {date_str}")

        current_date += timedelta(days=1)
        time.sleep(0.5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch UK Carbon Intensity Data")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    run_ingestion(args.start, args.end)