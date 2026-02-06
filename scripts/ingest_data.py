import argparse
import logging
import sys
import time
from pathlib import Path

# Add 'src' to the Python path so we can import our modules
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.append(str(PROJECT_ROOT / "src"))

# Import the logic we wrote in previous steps
from data_ingestion import fetch_carbon, fetch_weather, normalise, merge_sources
from utils import generate_lineage

# Configure global logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [PIPELINE] - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

def run_pipeline(start_date: str, end_date: str):
    total_start = time.time()
    logger.info(f"Starting Data Pipeline from {start_date} to {end_date}")
    
    # --- Step 1: Carbon Ingestion ---
    logger.info("--- Step 1/4: Ingesting Carbon Intensity Data ---")
    try:
        fetch_carbon.run_ingestion(start_date, end_date)
    except Exception as e:
        logger.error(f"Carbon Ingestion Failed: {e}")
        sys.exit(1)

    # --- Step 2: Weather Ingestion ---
    logger.info("--- Step 2/4: Ingesting Weather Data (Open-Meteo) ---")
    try:
        fetch_weather.run_ingestion(start_date, end_date)
    except Exception as e:
        logger.error(f"Weather Ingestion Failed: {e}")
        sys.exit(1)

    # --- Step 3: Normalisation ---
    logger.info("--- Step 3/4: Normalising Raw Data to Silver Tables ---")
    try:
        normalise.normalise_carbon()
        normalise.normalise_weather()
    except Exception as e:
        logger.error(f"Normalisation Failed: {e}")
        sys.exit(1)

    # --- Step 4: Merge & Validate ---
    logger.info("--- Step 4/4: Merging to Gold Dataset & Generating Metadata ---")
    try:
        merge_sources.merge_sources()
    except Exception as e:
        logger.error(f"Merge Failed: {e}")
        sys.exit(1)

    total_time = time.time() - total_start
    logger.info(f"Pipeline Completed Successfully in {total_time:.1f} seconds.")
    logger.info(f"Output: {PROJECT_ROOT}/data/processed/dataset_v1.parquet")

    # --- Step 5: Documentation ---
    logger.info("--- Step 5/5: Generating Data Lineage Diagram ---")
    try:
        generate_lineage.draw_lineage()
        logger.info("Lineage diagram updated in docs/data_lineage.png")
    except Exception as e:
        logger.warning(f"Could not generate diagram (is Graphviz installed?): {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Data Ingestion Pipeline")
    parser.add_argument("--start", required=True, help="Start Date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End Date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    run_pipeline(args.start, args.end)