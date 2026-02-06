import argparse
import sys
import yaml
import torch
import logging
import mlflow
import json
import shutil
import hashlib
from pathlib import Path
from datetime import datetime

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.append(str(PROJECT_ROOT / "src"))

from training.dataloaders import load_dataloaders
from models.tft import TemporalFusionTransformer
from training.train_tft import TFTTrainer
from utils.plotting import plot_forecast

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - [TRAIN] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def get_file_hash(path: Path) -> str:
    """Compute MD5 hash of a file for versioning."""
    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def main():
    parser = argparse.ArgumentParser(description="Train Carbon Forecasting Model")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    args = parser.parse_args()
    
    # 1. Config
    config_path = Path(args.config)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # 2. Data
    data_dir = PROJECT_ROOT / "data" / "training"
    try:
        train_loader, val_loader, _, meta = load_dataloaders(
            data_dir, config_path, batch_size=config['training']['batch_size']
        )
    except Exception as e:
        logger.error(f"Failed to load data: {e}"); sys.exit(1)

    # 3. Model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model = TemporalFusionTransformer(
        config=config,
        num_features=meta['num_features'],
        output_horizon=meta['horizon']
    )
    
    # 4. Trainer Setup
    mlflow.set_experiment(config['logging']['experiment_name'])
    save_dir = PROJECT_ROOT / "models" / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)
    trainer = TFTTrainer(config, model, device)
    
    # 5. Calculate Dataset Hash (Advanced Tracking)
    train_file_hash = get_file_hash(data_dir / "train.pt")
    
    logger.info(f"Starting MLflow Run...")
    
    with mlflow.start_run() as run:
        # A. Log Params & Data Version
        mlflow.log_params(config['model'])
        mlflow.log_params(config['training'])
        mlflow.log_param("dataset_hash", train_file_hash)  # data versioning
        mlflow.log_param("num_features", meta['num_features'])
        
        # B. Train
        try:
            trainer.fit(train_loader, val_loader, config['training']['max_epochs'], save_dir)
        except KeyboardInterrupt:
            logger.warning("Interrupted. Saving current state...")
            
        # C. Log Model (Safe Mode)
        best_model_path = save_dir / "best_model.pt"
        if best_model_path.exists():
            # Load best weights for plotting
            model.load_state_dict(torch.load(best_model_path, weights_only=True))
            
            logger.info("Logging best model...")
            try:
                mlflow.pytorch.log_model(model, "model", pip_requirements=[])
            except Exception as e:
                logger.error(f"Model logging failed: {e}")
                
            # D. Visual Sanity Check (Advanced Tracking)
            logger.info("Generating forecast plot...")
            try:
                fig = plot_forecast(model, val_loader, device)
                mlflow.log_figure(fig, "forecast_sanity_check.png")  # visualisation artifact
                logger.info("Logged forecast plot to MLflow")
            except Exception as e:
                logger.error(f"Plotting failed: {e}")
        else:
            logger.warning("No model to log.")

    # 6. Production Artifacts (Same as before)
    prod_dir = PROJECT_ROOT / "models" / "tft_prod"
    prod_dir.mkdir(parents=True, exist_ok=True)
    if best_model_path.exists():
        shutil.copy(best_model_path, prod_dir / "model.pt")
        
    metadata = {
        "model_type": "TemporalFusionTransformer",
        "version": "v1.1", # Bumped version
        "timestamp": datetime.utcnow().isoformat(),
        "dataset_hash": train_file_hash, # Track data version in prod metadata too
        "config": config['model'],
        "data_spec": {
            "num_features": meta['num_features'],
            "lookback": meta['lookback'],
            "horizon": meta['horizon']
        }
    }
    with open(prod_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    logger.info("Pipeline Complete.")

if __name__ == "__main__":
    main()