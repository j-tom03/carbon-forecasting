import matplotlib.pyplot as plt
import torch
import numpy as np
from typing import List

def plot_forecast(model, loader, device, quantiles: List[float] = [0.1, 0.5, 0.9]):
    """
    Runs inference on a single batch and plots the first sequence.
    Returns a Matplotlib Figure object for logging.
    """
    model.eval()
    
    # Get one batch
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)
    
    with torch.no_grad():
        # Shape: [batch, horizon, quantiles]
        preds = model(x)
    
    # Move to CPU for plotting
    y_true = y[0].cpu().numpy()          # Actual target (1st sample)
    preds = preds[0].cpu().numpy()       # Predictions (1st sample)
    
    # Indices for quantiles
    idx_10 = 0 # 0.1
    idx_50 = 1 # 0.5 (Median)
    idx_90 = 2 # 0.9
    
    # Create Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot Actual
    ax.plot(y_true, label="Actual (Scaled)", color="black", linewidth=2)
    
    # Plot Median Forecast
    ax.plot(preds[:, idx_50], label="Median Forecast", color="blue", linestyle="--")
    
    # Plot Uncertainty Interval (10th - 90th)
    ax.fill_between(
        range(len(y_true)),
        preds[:, idx_10],
        preds[:, idx_90],
        color="blue",
        alpha=0.2,
        label="Confidence Interval (10-90%)"
    )
    
    ax.set_title("Forecast Sanity Check (First Sequence in Validation Set)")
    ax.set_xlabel("Time Steps (Horizon)")
    ax.set_ylabel("Carbon Intensity (Scaled)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig