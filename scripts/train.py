import argparse
import sys
import yaml
import torch
import logging
import mlflow
import json
import shutil
from pathlib import Path
from datetime import datetime

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.append(str(PROJECT_ROOT / "src"))

from training.dataloaders import load_dataloaders
from models.tft import TemporalFusionTransformer
from training.train_tft import TFTTrainer

# Configure Logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - [TRAIN] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def main():
    """
    Main entry point for training the TFT model.
    Follows strict steps: Parse -> Load -> Train -> Log -> Exit.
    """
    
    # 1. Parse Config
    parser = argparse.ArgumentParser(description="Train Carbon Forecasting Model")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to the training configuration YAML file"
    )
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded configuration from {config_path}")

    # 2. Load Dataset
    # We assume the data artifact was built in the previous phase at 'data/training'
    data_dir = PROJECT_ROOT / "data" / "training"
    
    try:
        train_loader, val_loader, _, meta = load_dataloaders(
            data_dir, 
            config_path, 
            batch_size=config['training']['batch_size']
        )
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        sys.exit(1)

    # 3. Initialise Model
    # Determine hardware acceleration
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
    
    # 4. Setup MLflow & Trainer
    experiment_name = config['logging']['experiment_name']
    mlflow.set_experiment(experiment_name)
    
    # Directory to save local checkpoints before logging them
    save_dir = PROJECT_ROOT / "models" / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    trainer = TFTTrainer(config, model, device)
    
    logger.info(f"Starting MLflow Run: {experiment_name}")
    
    # 5. Execute Training Loop
    with mlflow.start_run() as run:
        # A. Log Parameters
        mlflow.log_params(config['model'])
        mlflow.log_params(config['training'])
        mlflow.log_param("num_features", meta['num_features'])
        mlflow.log_param("data_version", "v1") # Could be dynamic in future
        
        # B. Run Fit (Includes Epoch Loop & Validation)
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
        except Exception as e:
            logger.error(f"Training failed: {e}")
            sys.exit(1)
            
        # C. Log Final Artifacts
        # The trainer saves 'best_model.pt'. We log it to MLflow now.
        best_model_path = save_dir / "best_model.pt"
        if best_model_path.exists():
            logger.info("Logging best model artifact to MLflow...")
            mlflow.pytorch.log_model(model, "model")
        else:
            logger.warning("No best model file found to log.")

    logger.info("Generating production artifacts...")
    
    prod_dir = PROJECT_ROOT / "models" / "tft_prod"
    prod_dir.mkdir(parents=True, exist_ok=True)
    
    # A. Save Weights (Clean Copy)
    final_model_path = prod_dir / "model.pt"
    if best_model_path.exists():
        shutil.copy(best_model_path, final_model_path)
        logger.info(f"Saved production model: {final_model_path}")
    else:
        logger.error("Training failed to produce a model checkpoint.")
        sys.exit(1)

    # B. Save Metadata (JSON Sidecar)
    # The API will read this to know how to configure the model class
    metadata = {
        "model_type": "TemporalFusionTransformer",
        "version": "v1.0",
        "timestamp": datetime.utcnow().isoformat(),
        "config": {
            "hidden_size": config['model']['hidden_size'],
            "dropout": config['model']['dropout'],
            "attention_heads": config['model']['attention_heads'],
            "quantiles": config['loss']['quantiles']
        },
        "data_spec": {
            "num_features": meta['num_features'],
            "lookback": meta['lookback'],
            "horizon": meta['horizon'],
            "feature_names": meta.get('feature_names', [])
        }
    }
    
    meta_path = prod_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
        
    logger.info(f"Saved inference metadata: {meta_path}")
    logger.info("Training pipeline completed successfully.")

if __name__ == "__main__":
    main()
