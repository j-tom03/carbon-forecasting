import torch
from torch.utils.data import DataLoader
import logging
import yaml
from pathlib import Path
from typing import Tuple, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [DATALOADERS] - %(message)s")
logger = logging.getLogger(__name__)

import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(PROJECT_ROOT / "src"))
from training.dataset_builder import TimeSeriesDataset

def load_dataloaders(
    data_dir: Path, 
    config_path: Path,
    batch_size: int = 64,
    num_workers: int = 0  # Set to 0 for debugging, 2-4 for production
    ) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Loads .pt dataset artifacts and wraps them in PyTorch DataLoaders.
    
    Args:
        data_dir: Path to directory containing train.pt, val.pt, meta.yaml
        config_path: Path to train_tft.yaml (to verify consistency)
        batch_size: Batch size for training
        
    Returns:
        train_loader, val_loader, test_loader, metadata
    """
    logger.info(f"Loading datasets from {data_dir}...")
    
    # 1. Load Metadata & Config
    if not (data_dir / "meta.yaml").exists():
        raise FileNotFoundError(f"Metadata not found in {data_dir}")
        
    with open(data_dir / "meta.yaml", "r") as f:
        meta = yaml.safe_load(f)
        
    with open(config_path, "r") as f:
        train_config = yaml.safe_load(f)
        
    # 2. Critical Checks (Assertions)
    if "horizon" in meta:
        assert meta["horizon"] == meta["horizon"], "Meta horizon mismatch (logic check)"
    
    logger.info(f"Dataset Metadata: {meta['num_features']} features, Lookback {meta['lookback']}, Horizon {meta['horizon']}")

    # 3. Load Artifacts
    datasets = {}
    for split in ["train", "val", "test"]:
        file_path = data_dir / f"{split}.pt"
        if not file_path.exists():
            raise FileNotFoundError(f"Missing {split} dataset at {file_path}")
            
        # torch.load deserializes the TimeSeriesDataset object
        datasets[split] = torch.load(file_path, weights_only=False)
        logger.info(f"Loaded {split}: {len(datasets[split])} samples")

    # 4. Create DataLoaders
    # shuffle=True only for training to break correlations
    train_loader = DataLoader(
        datasets["train"], 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        datasets["val"], 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        datasets["test"], 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )

    # 5. Sanity Check: Inspect one batch
    try:
        x_batch, y_batch = next(iter(train_loader))
        
        # Check shapes
        # X: [batch, lookback, features]
        # Y: [batch, horizon]
        assert x_batch.dim() == 3, f"Expected 3D input (Batch, Time, Feat), got {x_batch.shape}"
        assert y_batch.dim() == 2, f"Expected 2D target (Batch, Time), got {y_batch.shape}"
        
        # Check for NaNs
        if torch.isnan(x_batch).any() or torch.isnan(y_batch).any():
            raise ValueError("CRITICAL: NaNs detected in training batch! Check preprocessing.")
            
        logger.info(f"Sanity Check Passed. Batch Shape: X={x_batch.shape}, Y={y_batch.shape}")
        
    except Exception as e:
        logger.error(f"Sanity check failed: {e}")
        raise e

    return train_loader, val_loader, test_loader, meta

if __name__ == "__main__":
    # Test Harness
    # Assumes you have run the builder and have data in data/training
    DATA_DIR = PROJECT_ROOT / "data" / "training"
    CONFIG_PATH = PROJECT_ROOT / "configs" / "train_tft.yaml"
    
    if DATA_DIR.exists():
        load_dataloaders(DATA_DIR, CONFIG_PATH)
    else:
        print("Please run dataset_builder.py first.")