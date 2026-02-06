import optuna
import mlflow
import copy
from typing import Dict, Any
import torch

from training.train_tft import train_and_evaluate

class TFTObjective:
    """
    The Objective Function for Optuna.
    It holds the fixed context (Data, Device, Base Config) and 
    runs one training loop per trial with sampled hyperparameters.
    """
    def __init__(
        self, 
        base_config: Dict[str, Any], 
        train_loader: Any, 
        val_loader: Any, 
        meta: Dict[str, Any], 
        device: torch.device
    ):
        self.base_config = base_config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.meta = meta
        self.device = device

    def __call__(self, trial: optuna.Trial) -> float:
        # 1. Sample Hyperparameters (The Grid)
        lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        hidden_size = trial.suggest_categorical("hidden_size", [16, 32, 64])
        dropout = trial.suggest_float("dropout", 0.0, 0.2, step=0.1) # 0.0, 0.1, 0.2
        
        # 2. Update Configuration
        config = copy.deepcopy(self.base_config)
        config['training']['learning_rate'] = lr
        config['model']['hidden_size'] = hidden_size
        config['model']['dropout'] = dropout
        
        # 3. MLflow Tracking (One run per trial)
        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):  # nested=True allows these runs to be grouped under a parent "Tuning" run
            # Log params for this specific trial
            mlflow.log_params({
                "learning_rate": lr,
                "hidden_size": hidden_size,
                "dropout": dropout,
                "trial_id": trial.number
            })
            
            # Tag the run for easy filtering later
            mlflow.set_tag("trial_id", str(trial.number))
            mlflow.set_tag("stage", "tuning")
            
            # 4. Train and Evaluate
            try:
                val_loss = train_and_evaluate(
                    config=config,
                    train_loader=self.train_loader,
                    val_loader=self.val_loader,
                    meta=self.meta,
                    device=self.device
                )
                
                # Log the result
                mlflow.log_metric("val_loss", val_loss)
                return val_loss
                
            except Exception as e:
                # If a trial fails prune it instead of crashing the whole study
                print(f"Trial {trial.number} failed: {e}")
                # Return infinity so Optuna knows this was a bad run
                return float('inf')