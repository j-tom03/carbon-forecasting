import torch

def quantile_loss(preds: torch.Tensor, target: torch.Tensor, quantiles: list) -> torch.Tensor:
    """
    Calculates the Quantile Loss (Pinball Loss) for probabilistic regression.
    
    Args:
        preds: Predictions of shape [batch_size, horizon, num_quantiles]
        target: Actual values of shape [batch_size, horizon]
        quantiles: List of quantiles to evaluate (e.g., [0.1, 0.5, 0.9])
        
    Returns:
        torch.Tensor: Scalar loss value (mean over batch and time)
    """
    losses = []
    
    # Expand target to match prediction shape: [batch, horizon, 1]
    target = target.unsqueeze(-1)
    
    for i, q in enumerate(quantiles):
        # Extract the prediction for this specific quantile
        pred_q = preds[:, :, i:i+1]
        
        # Calculate error
        errors = target - pred_q
        
        # Pinball Loss Formula: max((q-1)*error, q*error)
        loss = torch.max((q - 1) * errors, q * errors)
        losses.append(loss)
        
    # Stack losses and average them
    # We sum over quantiles, then mean over batch and time dimensions
    total_loss = torch.stack(losses, dim=2).sum(dim=2).mean()
    
    return total_loss