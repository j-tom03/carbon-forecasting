import torch
import torch.nn as nn
from typing import Dict

class TemporalFusionTransformer(nn.Module):
    """
    A simplified implementation of the Temporal Fusion Transformer architecture.
    
    It uses an LSTM to encode temporal patterns and Multi-Head Attention to 
    capture long-term dependencies, producing probabilistic forecasts.
    """
    def __init__(self, config: Dict, num_features: int, output_horizon: int):
        super().__init__()
        
        # Configuration parameters
        hidden_size = config['model']['hidden_size']
        dropout = config['model']['dropout']
        heads = config['model']['attention_heads']
        quantiles = config['loss']['quantiles']
        
        self.output_horizon = output_horizon
        self.num_quantiles = len(quantiles)

        # 1. Input Projection
        # Projects continuous features to the internal hidden size
        self.feature_projector = nn.Linear(num_features, hidden_size)
        
        # 2. Sequence Encoder (LSTM)
        # Processes the input sequence to create a context vector
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=dropout if config['model'].get('lstm_layers', 1) > 1 else 0
        )
        
        # 3. Temporal Attention Mechanism
        # Allows the model to look back at specific relevant past time steps
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Normalisation layer for residual connection
        self.post_attn_norm = nn.LayerNorm(hidden_size)
        
        # 4. Output Decoder
        # Maps the hidden state to the desired output shape (Horizon x Quantiles)
        self.output_layer = nn.Linear(hidden_size, self.num_quantiles)
        
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape [batch_size, lookback, num_features]
            
        Returns:
            Tensor of shape [batch_size, horizon, num_quantiles]
        """
        # 1. Project inputs
        x_emb = self.feature_projector(x) # [batch, lookback, hidden]
        
        # 2. Encode sequence
        # We use the full sequence for attention keys/values
        lstm_out, (h_n, c_n) = self.lstm(x_emb) # [batch, lookback, hidden]
        
        # 3. Attention
        # Query: The most recent hidden state (what is happening now?)
        # Key/Value: The entire history (what happened before?)
        
        # Reshape last state to [batch, 1, hidden] to act as the query
        last_state = lstm_out[:, -1, :].unsqueeze(1)
        
        # Apply attention
        attn_out, _ = self.attention(query=last_state, key=lstm_out, value=lstm_out)
        
        # Residual connection and normalisation
        context = self.post_attn_norm(last_state + attn_out) # [batch, 1, hidden]
        
        # 4. Decode
        # Expand context to cover the entire forecast horizon
        context_expanded = context.repeat(1, self.output_horizon, 1) # [batch, horizon, hidden]
        
        # Generate predictions for all quantiles
        predictions = self.output_layer(context_expanded) # [batch, horizon, quantiles]
        
        return predictions