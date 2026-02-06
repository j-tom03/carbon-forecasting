import argparse
import sys
import yaml
import torch
import logging
import mlflow
import optuna
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.append(str(PROJECT_ROOT / "src"))

from training.dataloaders import load_dataloaders
from training.optuna_objective import TFTObjective

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

    # 1. Load Configuration
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)
        
    with open(config_path, "r") as f:
        base_config = yaml.safe_load(f)

    # 2. Setup MLflow
    experiment_name = base_config['logging']['experiment_name']
    mlflow.set_tracking_uri(f"file://{PROJECT_ROOT}/mlruns")
    mlflow.set_experiment(experiment_name)
    
    # 3. Load Data ONCE
    logger.info("Loading data into memory...")
    data_dir = PROJECT_ROOT / "data" / "training"
    
    try:
        train_loader, val_loader, _, meta = load_dataloaders(
            data_dir, 
            config_path, 
            batch_size=base_config['training']['batch_size']
        )
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Tuning on device: {device}")

    # 4. Initialise Objective
    objective = TFTObjective(
        base_config=base_config,
        train_loader=train_loader,
        val_loader=val_loader,
        meta=meta,
        device=device
    )

    # 5. Run Optimisation
    logger.info(f"Starting HPO Campaign for {args.n_trials} trials...")
    
    # Start a "Parent Run" to group the trials in the UI
    with mlflow.start_run(run_name=f"HPO_Campaign_{args.n_trials}_trials") as parent_run:
        mlflow.set_tag("type", "hpo_parent")
        mlflow.set_tag("search_space", "v1")
        
        # Create Optuna Study
        study = optuna.create_study(direction="minimize")
        
        # Execute Trials
        try:
            study.optimize(objective, n_trials=args.n_trials)
        except KeyboardInterrupt:
            logger.warning("HPO interrupted by user. Saving current progress...")

        # 6. Log Best Results
        logger.info("------------------------------------------------")
        logger.info(f"Best Trial: #{study.best_trial.number}")
        logger.info(f"Best Loss: {study.best_value:.4f}")
        logger.info("Best Params:")
        for key, value in study.best_params.items():
            logger.info(f"  {key}: {value}")
            
        # Log best params to the parent run for easy visibility
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_val_loss", study.best_value)
        
        logger.info("------------------------------------------------")
        logger.info("HPO Complete. Check MLflow for details.")

if __name__ == "__main__":
    main()