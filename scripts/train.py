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
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except Exception:
        return "unknown"

def get_file_hash(path: Path) -> str:
    hash_md5 = hashlib.md5()
    if not path.exists(): return "unknown"
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def main():
    # 1. Parse Config
    parser = argparse.ArgumentParser(description="Train Carbon Forecasting Model")
    parser.add_argument("--config", type=str, required=True, help="Path to the training config")
    args = parser.parse_args()
    
    config_path = Path(args.config)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # 2. Setup MLflow
    mlflow.set_tracking_uri(f"file://{PROJECT_ROOT}/mlruns")
    mlflow.set_experiment(config['logging']['experiment_name'])
    
    # 3. Load Data & Context
    data_dir = PROJECT_ROOT / "data" / "training"
    train_loader, val_loader, _, meta = load_dataloaders(
        data_dir, 
        config_path, 
        batch_size=config['training']['batch_size']
    )
    
    git_hash = get_git_commit()
    data_hash = get_file_hash(data_dir / "train.pt")
    
    # 4. Initialise Model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = TemporalFusionTransformer(
        config=config,
        num_features=meta['num_features'],
        output_horizon=meta['horizon']
    )
    
    save_dir = PROJECT_ROOT / "models" / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)
    trainer = TFTTrainer(config, model, device)
    
    # 5. RUN TRAINING
    with mlflow.start_run() as run:
        
        # Setting tags for automation
        mlflow.set_tags({
            "model_type": "TFT",
            "dataset_version": "v1",
            "forecast_horizon": meta['horizon'],
            "stage": "experimental",
            "git_commit": git_hash
        })
        
        # Log Params
        mlflow.log_params(config['model'])
        mlflow.log_params(config['training'])
        mlflow.log_param("dataset_hash", data_hash)
        mlflow.log_artifact(str(config_path), artifact_path="config")

        # Train
        try:
            trainer.fit(train_loader, val_loader, config['training']['max_epochs'], save_dir)
        except KeyboardInterrupt:
            logger.warning("Interrupted. Saving current state...")

        # Log artifacts
        best_model_path = save_dir / "best_model.pt"
        if best_model_path.exists():
            # A. Log Model Binary
            model.load_state_dict(torch.load(best_model_path, weights_only=True))
            mlflow.pytorch.log_model(model, "model", pip_requirements=[])
            
            # B. Log Inference Metadata (JSON)
            metadata = {
                "model_type": "TemporalFusionTransformer",
                "version": "v1.3",
                "timestamp": datetime.utcnow().isoformat(),
                "config": config['model'],
                "data_spec": {
                    "num_features": meta['num_features'],
                    "lookback": meta['lookback'],
                    "horizon": meta['horizon'],
                    "feature_names": meta.get('feature_names', [])  # defaults to empty
                }
            }
            
            # Save locally first
            local_meta_path = save_dir / "metadata.json"
            with open(local_meta_path, "w") as f:
                json.dump(metadata, f, indent=4)
            
            # Log to MLflow
            mlflow.log_artifact(str(local_meta_path), artifact_path="inference_context")
            logger.info("Logged metadata.json to MLflow")

            # C. Log Sanity Plot
            try:
                fig = plot_forecast(model, val_loader, device)
                mlflow.log_figure(fig, "forecast_sanity_check.png")
            except Exception as e:
                logger.error(f"Plotting failed: {e}")
                
            # D. Create Local Production Folder (For API dev)
            prod_dir = PROJECT_ROOT / "models" / "tft_prod"
            prod_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(best_model_path, prod_dir / "model.pt")
            shutil.copy(local_meta_path, prod_dir / "metadata.json")
            logger.info(f"Production artifacts ready in {prod_dir}")
            
        else:
            logger.error("No model saved.")
            sys.exit(1)

if __name__ == "__main__":
    main()