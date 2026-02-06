import argparse
import sys
import yaml
import torch
import logging
import mlflow
import json
import shutil
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.append(str(PROJECT_ROOT / "src"))

from training.dataloaders import load_dataloaders
from models.tft import TemporalFusionTransformer
from training.train_tft import TFTTrainer
from utils.plotting import plot_forecast

# Configure Logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - [TRAIN] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def get_git_commit() -> str:
    """Returns the current git commit hash to track code version."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except Exception:
        return "unknown"

def get_file_hash(path: Path) -> str:
    """Compute MD5 hash of a file to track data version."""
    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def main():
    """
    Main entry point for training.
    Implements strict MLOps tracking: Git + Data Hash + Config Artifact.
    """
    
    # 1. Parse Config
    parser = argparse.ArgumentParser(description="Train Carbon Forecasting Model")
    parser.add_argument("--config", type=str, required=True, help="Path to the training config")
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # 2. Initialise MLflow EARLY
    # We set this before loading data or models to ensure everything is captured
    experiment_name = config['logging']['experiment_name']
    mlflow.set_tracking_uri(f"file://{PROJECT_ROOT}/mlruns")
    mlflow.set_experiment(experiment_name)
    logger.info(f"Initialized MLflow Experiment: {experiment_name}")

    # 3. Load Context (Data & Versioning)
    data_dir = PROJECT_ROOT / "data" / "training"
    
    # Calculate Context Hashes BEFORE training
    git_hash = get_git_commit()
    data_hash = get_file_hash(data_dir / "train.pt")
    logger.info(f"Run Context -> Git: {git_hash[:7]} | Data: {data_hash[:7]}")

    # Load Data
    try:
        train_loader, val_loader, _, meta = load_dataloaders(
            data_dir, 
            config_path, 
            batch_size=config['training']['batch_size']
        )
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        sys.exit(1)

    # 4. Initialise Model
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Training on device: {device}")

    model = TemporalFusionTransformer(
        config=config,
        num_features=meta['num_features'],
        output_horizon=meta['horizon']
    )
    
    # Directory for checkpoints
    save_dir = PROJECT_ROOT / "models" / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    trainer = TFTTrainer(config, model, device)
    
    # 5. EXECUTE RUN (With Full Context Logging)
    with mlflow.start_run() as run:
        
        # A. LOG IMMUTABLE CONTEXT
        mlflow.log_params(config['model'])
        mlflow.log_params(config['training'])
        
        # Critical MLOps Metadata
        mlflow.log_param("git_commit", git_hash)
        mlflow.log_param("dataset_hash", data_hash)
        mlflow.log_param("num_features", meta['num_features'])
        mlflow.log_param("horizon", meta['horizon'])
        
        # Save the actual Config File as an artifact (Exact Reproducibility)
        mlflow.log_artifact(str(config_path), artifact_path="config")
        
        # B. TRAIN
        try:
            trainer.fit(
                train_loader, 
                val_loader, 
                epochs=config['training']['max_epochs'], 
                save_dir=save_dir
            )
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user. Saving state...")
            sys.exit(0)
            
        # C. LOG ARTIFACTS
        best_model_path = save_dir / "best_model.pt"
        if best_model_path.exists():
            # 1. Log Model Binary
            logger.info("Logging best model artifact...")
            try:
                # Load best weights to ensure plot matches best model
                model.load_state_dict(torch.load(best_model_path, weights_only=True))
                
                mlflow.pytorch.log_model(
                    model, 
                    "model", 
                    pip_requirements=[] # Safe Mode for Python 3.11/3.13
                )
            except Exception as e:
                logger.error(f"MLflow model logging failed: {e}")

            # 2. Log Visual Sanity Check
            logger.info("Generating forecast plot...")
            try:
                fig = plot_forecast(model, val_loader, device)
                mlflow.log_figure(fig, "forecast_sanity_check.png")
                logger.info("Logged forecast plot")
            except Exception as e:
                logger.error(f"Plotting failed: {e}")
        else:
            logger.warning("No best model file found to log.")

    # 6. Production Handoff (The "Artifact" Step)
    logger.info("Generating production artifacts...")
    prod_dir = PROJECT_ROOT / "models" / "tft_prod"
    prod_dir.mkdir(parents=True, exist_ok=True)
    
    if best_model_path.exists():
        shutil.copy(best_model_path, prod_dir / "model.pt")
        
        # Create Production Metadata (Sidecar)
        metadata = {
            "model_type": "TemporalFusionTransformer",
            "version": "v1.2", 
            "timestamp": datetime.utcnow().isoformat(),
            "git_commit": git_hash,
            "dataset_hash": data_hash,
            "config": config['model'],
            "data_spec": {
                "num_features": meta['num_features'],
                "horizon": meta['horizon']
            }
        }
        
        with open(prod_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)
            
        logger.info(f"Saved production artifacts to {prod_dir}")
    else:
        logger.error("Failed to generate production artifacts.")
        sys.exit(1)

if __name__ == "__main__":
    main()