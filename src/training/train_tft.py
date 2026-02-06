import torch
import torch.nn as nn
import torch.optim as optim
import logging
import time
import mlflow
import numpy as np
from pathlib import Path
from typing import Dict, Any

from models.tft import TemporalFusionTransformer
from models.loss import quantile_loss

logger = logging.getLogger(__name__)

class TFTTrainer:
    def __init__(self, config: Dict, model: TemporalFusionTransformer, device: torch.device):
        self.config = config
        self.model = model.to(device)
        self.device = device
        
        # Setup Optimiser
        lr = float(config['training']['learning_rate'])
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Setup Quantiles
        self.quantiles = config['loss']['quantiles']
        
    def train_epoch(self, loader) -> float:
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad()
            preds = self.model(x) 
            loss = quantile_loss(preds, y, self.quantiles)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(loader)

    def validate(self, loader) -> Dict[str, float]:
        """
        Runs validation and calculates:
        1. Quantile Loss (Optimization Target)
        2. MAE (Interpretable Accuracy)
        3. Coverage (Calibration Check)
        """
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        total_coverage = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                
                # preds shape: [batch, horizon, 3] (0.1, 0.5, 0.9)
                preds = self.model(x)
                
                # 1. Quantile Loss
                loss = quantile_loss(preds, y, self.quantiles)
                total_loss += loss.item()
                
                # 2. MAE on Median Forecast (Index 1 is the 0.5 quantile)
                # y shape: [batch, horizon]
                median_preds = preds[:, :, 1]
                mae = torch.abs(median_preds - y).mean()
                total_mae += mae.item()
                
                # 3. Coverage (Did y fall between 0.1 and 0.9?)
                # Index 0 is 0.1 (Lower), Index 2 is 0.9 (Upper)
                lower = preds[:, :, 0]
                upper = preds[:, :, 2]
                
                # Create a boolean mask: 1 if inside, 0 if outside
                inside_interval = (y >= lower) & (y <= upper)
                coverage = inside_interval.float().mean()
                total_coverage += coverage.item()
                
                num_batches += 1
                
        metrics = {
            "val_loss": total_loss / num_batches,
            "val_mae": total_mae / num_batches,
            "val_coverage": total_coverage / num_batches
        }
        return metrics

    def fit(self, train_loader, val_loader, epochs: int, save_dir: Path):
        best_val_loss = float('inf')
        
        logger.info(f"Starting training for {epochs} epochs on {self.device}...")
        
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate (Now returns a Dict of metrics)
            val_metrics = self.validate(val_loader)
            val_loss = val_metrics["val_loss"]
            
            duration = time.time() - start_time
            
            # Explicitly log all 3 metrics per epoch
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mae": val_metrics["val_mae"],           # Accuracy
                "val_coverage": val_metrics["val_coverage"], # Calibration
                "epoch_time": duration
            }, step=epoch)
            
            logger.info(
                f"Epoch {epoch} | "
                f"Loss: {train_loss:.4f} / {val_loss:.4f} | "
                f"MAE: {val_metrics['val_mae']:.4f} | "
                f"Cov: {val_metrics['val_coverage']:.1%}" # e.g., 85.0%
            )
            
            # Save Best Model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = save_dir / "best_model.pt"
                torch.save(self.model.state_dict(), save_path)
                logger.info(f"New best model saved")

        # Load best weights back for final artifacts
        best_path = save_dir / "best_model.pt"
        if best_path.exists():
            self.model.load_state_dict(torch.load(best_path, weights_only=True))


def train_and_evaluate(
    config: Dict[str, Any],
    train_loader: Any,
    val_loader: Any,
    meta: Dict[str, Any],
    device: torch.device,
    enable_logging: bool = True) -> float:
    """
    Standard interface for training. Used by both the main training script
    and the Optuna tuning script.
    
    Returns:
        float: The final validation loss (metric to minimize).
    """
    # 1. Initialise Model with the specific config for this run
    # (Optuna will inject different hidden_sizes/dropouts here)
    model = TemporalFusionTransformer(
        config=config,
        num_features=meta['num_features'],
        output_horizon=meta['horizon']
    )
    
    # 2. Setup Trainer
    trainer = TFTTrainer(config, model, device)
    
    # 3. Train
    # We use a temporary directory for checkpoints to avoid cluttering 
    # the main models/ folder during tuning trials.
    temp_dir = Path("models/temp_tuning")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    trainer.fit(
        train_loader, 
        val_loader, 
        epochs=config['training']['max_epochs'], 
        save_dir=temp_dir
    )
    
    # 4. Calculate Final Metric
    val_metrics = trainer.validate(val_loader)
    final_loss = val_metrics["val_loss"]
    
    return final_loss