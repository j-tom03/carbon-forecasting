import torch
from torchview import draw_graph
import sys
from pathlib import Path
import yaml

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(PROJECT_ROOT / "src"))

from models.tft import TemporalFusionTransformer

def generate_model_diagram():
    # 1. Load Config
    config_path = PROJECT_ROOT / "configs" / "train_tft.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # 2. Instantiate Model
    # Use dummy dimensions matching your data
    model = TemporalFusionTransformer(config, num_features=20, output_horizon=96)
    
    # 3. Generate Graph
    # input_size = (Batch Size, Lookback, Features)
    # depth=2 keeps it high-level (shows 'LSTM', 'MultiheadAttention' blocks)
    # expand_nested=True lets us see inside the main class
    print("Generating graph...")
    model_graph = draw_graph(
        model, 
        input_size=(1, 96, 20), 
        expand_nested=True,
        depth=2,
        save_graph=True,
        filename="model_architecture",
        directory=str(PROJECT_ROOT / "docs")
    )
    
    print(f"Model diagram saved to docs/model_architecture.png")

if __name__ == "__main__":
    generate_model_diagram()