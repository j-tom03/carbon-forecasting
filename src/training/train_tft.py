import torch
import torch.nn as nn
import torch.optim as optim
import logging
import time
import mlflow
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
        
        # Setup Optimizer
        lr = float(config['training']['learning_rate'])
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Setup Quantiles for Loss
        self.quantiles = config['loss']['quantiles']
        
    def train_epoch(self, loader) -> float:
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(self.device), y.to(self.device)
            
            # 1. Zero Gradients
            self.optimizer.zero_grad()
            
            # 2. Forward Pass
            # x shape: [batch, lookback, features]
            preds = self.model(x) 
            
            # 3. Compute Loss
            # preds shape: [batch, horizon, quantiles]
            # y shape: [batch, horizon]
            loss = quantile_loss(preds, y, self.quantiles)
            
            # 4. Backward Pass
            loss.backward()
            
            # 5. Update Weights
            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(loader)

    def validate(self, loader) -> float:
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                preds = self.model(x)
                loss = quantile_loss(preds, y, self.quantiles)
                total_loss += loss.item()
                
        return total_loss / len(loader)

    def fit(self, train_loader, val_loader, epochs: int, save_dir: Path):
        best_val_loss = float('inf')
        
        logger.info(f"Starting training for {epochs} epochs on {self.device}...")
        
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            
            # Run cycles
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            duration = time.time() - start_time

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epoch_time": duration
            }, step=epoch)
            
            # Logging
            logger.info(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Time: {duration:.1f}s"
            )
            
            # Save Best Model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = save_dir / "best_model.pt"
                torch.save(self.model.state_dict(), save_path)
                logger.info(f"New best model saved to {save_path}")

        # Load the best weights back before logging to MLflow
        best_model_path = save_dir / "best_model.pt"
        if best_model_path.exists():
            self.model.load_state_dict(torch.load(best_model_path))
            
            # Log the model to MLflow Model Registry
            logger.info("Logging best model to MLflow...")
            mlflow.pytorch.log_model(self.model, "model")