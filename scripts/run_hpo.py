import argparse
import sys
import yaml
import torch
import logging
import mlflow
import optuna
import json
import shutil
import copy
from pathlib import Path
from datetime import datetime

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.append(str(PROJECT_ROOT / "src"))

from training.dataloaders import load_dataloaders
from training.optuna_objective import TFTObjective
from training.train_tft import train_and_evaluate

# Configure Logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - [HPO] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run Hyperparameter Optimisation")
    parser.add_argument("--config", type=str, default="configs/train_tft.yaml", help="Path to base config")
    parser.add_argument("--n_trials", type=int, default=10, help="Number of trials to run")
    args = parser.parse_args()

    # 1. Load Configuration & Data
    config_path = Path(args.config)
    with open(config_path, "r") as f:
        base_config = yaml.safe_load(f)

    mlflow.set_tracking_uri(f"file://{PROJECT_ROOT}/mlruns")
    mlflow.set_experiment(base_config['logging']['experiment_name'])
    
    logger.info("Loading data into memory...")
    data_dir = PROJECT_ROOT / "data" / "training"
    train_loader, val_loader, _, meta = load_dataloaders(
        data_dir, config_path, batch_size=base_config['training']['batch_size']
    )
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # 2. Run Optimisation
    objective = TFTObjective(base_config, train_loader, val_loader, meta, device)
    
    logger.info(f"Starting HPO Campaign for {args.n_trials} trials...")
    
    with mlflow.start_run(run_name=f"HPO_Campaign_{args.n_trials}_trials") as parent_run:
        mlflow.set_tag("type", "hpo_parent")
        
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=args.n_trials)
        
        # Log Best Results
        best_params = study.best_params
        mlflow.log_params(best_params)
        mlflow.log_metric("best_val_loss", study.best_value)
        
        logger.info("------------------------------------------------")
        logger.info(f"Best Trial: Loss {study.best_value:.4f}")
        logger.info(f"Params: {best_params}")
        logger.info("------------------------------------------------")

        # Automated Promotion
        logger.info("Promoting Best Model to Production...")
        
        # A. Create "Golden Config" from best params
        final_config = copy.deepcopy(base_config)
        final_config['training']['learning_rate'] = best_params['learning_rate']
        final_config['model']['hidden_size'] = best_params['hidden_size']
        final_config['model']['dropout'] = best_params['dropout']
        
        # B. Retrain the winner (Log as a distinct "Production Candidate" run)
        with mlflow.start_run(nested=True, run_name="Production_Candidate_Retrain"):
            mlflow.set_tag("stage", "production_candidate")
            mlflow.log_params(best_params)
            
            prod_dir = PROJECT_ROOT / "models" / "tft_prod"
            
            # This saves the model directly to the production folder
            final_loss = train_and_evaluate(
                config=final_config,
                train_loader=train_loader,
                val_loader=val_loader,
                meta=meta,
                device=device,
                save_dir=prod_dir 
            )
            
            logger.info(f"Production model retrained (Loss: {final_loss:.4f})")
            
            # C. Generate Metadata (Critical for API)
            metadata = {
                "model_type": "TemporalFusionTransformer",
                "version": "v2.0-hpo-optimized",
                "timestamp": datetime.utcnow().isoformat(),
                "config": final_config['model'], # Save the optimized config
                "metrics": {"val_loss": final_loss},
                "data_spec": {
                    "num_features": meta['num_features'],
                    "horizon": meta['horizon']
                }
            }
            
            with open(prod_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=4)
                
            logger.info(f"Artifacts promoted to: {prod_dir}")

if __name__ == "__main__":
    main()