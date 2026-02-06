import argparse
import sys
from pathlib import Path

# Add src to path so we can import the training module
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.append(str(PROJECT_ROOT / "src"))

from training import dataset_builder

def main():
    parser = argparse.ArgumentParser(description="Build Carbon Forecasting Dataset")
    
    # Argument matching the screenshot instructions
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to the dataset configuration YAML file"
    )
    
    args = parser.parse_args()
    
    config_path = Path(args.config)
    
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
        
    # Execute the builder
    dataset_builder.build_dataset(config_path)

if __name__ == "__main__":
    main()