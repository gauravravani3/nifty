import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :].unsqueeze(0)
        return x

class BranchTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super(BranchTransformer, self).__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

    def forward(self, src):
        # src shape: (batch_size, seq_len, input_dim)
        x = self.input_linear(src)
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x)
        # Take the last sequence output
        last_out = output[:, -1, :]
        return last_out

class UniversalTradingTransformer(nn.Module):
    def __init__(self, num_features_1m=36, num_features_5m=36, num_regimes=8, d_model=64):
        super(UniversalTradingTransformer, self).__init__()

        # Branch 1: 1-Minute Data
        self.branch_1m = BranchTransformer(input_dim=num_features_1m, d_model=d_model, nhead=4, num_layers=2)

        # Branch 2: 5-Minute Data
        self.branch_5m = BranchTransformer(input_dim=num_features_5m, d_model=d_model, nhead=4, num_layers=2)

        # Branch 3: HMM Regime Embedding
        self.regime_embedding = nn.Embedding(num_embeddings=num_regimes, embedding_dim=16)

        # Combined Feature Dimension
        combined_dim = (d_model * 2) + 16

        # Shared Fully Connected Layers
        self.fc_shared = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # 6 Independent Output Heads

        # Head 1: Direction (Binary: Target Hit = 1, StopLoss/Time Exit = 0)
        self.head_direction = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Head 2: Time to Event (Regression: 1 to 15 candles)
        self.head_time_to_event = nn.Sequential(
            nn.Linear(64, 1),
            nn.ReLU() # Time can't be negative
        )

        # Head 3: Optimal Entry Delay (Regression: 0 to 2 candles)
        self.head_entry_delay = nn.Sequential(
            nn.Linear(64, 1),
            nn.ReLU()
        )

        # Head 4: Optimal ATR Multiplier for SL (Regression)
        self.head_atr_mult = nn.Sequential(
            nn.Linear(64, 1),
            nn.ReLU()
        )

        # Head 5: Confidence Score of Direction
        self.head_confidence = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Head 6: Move Strength Classification (Strong vs Weak)
        self.head_move_strength = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, seq_1m, seq_5m, regime_state):
        # seq_1m: (batch, seq_len_1m, 36)
        # seq_5m: (batch, seq_len_5m, 36)
        # regime_state: (batch)

        out_1m = self.branch_1m(seq_1m)
        out_5m = self.branch_5m(seq_5m)
        regime_emb = self.regime_embedding(regime_state)

        # Concatenate branches
        combined = torch.cat((out_1m, out_5m, regime_emb), dim=1)

        # Shared features
        shared_features = self.fc_shared(combined)

        # Compute Heads
        direction = self.head_direction(shared_features)
        time_to_event = self.head_time_to_event(shared_features)
        entry_delay = self.head_entry_delay(shared_features)
        atr_mult = self.head_atr_mult(shared_features)
        confidence = self.head_confidence(shared_features)
        move_strength = self.head_move_strength(shared_features)

        return {
            'direction': direction,
            'time_to_event': time_to_event,
            'entry_delay': entry_delay,
            'atr_mult': atr_mult,
            'confidence': confidence,
            'move_strength': move_strength
        }

if __name__ == "__main__":
    # Quick shape test
    model = UniversalTradingTransformer()
    print("Model initialized. Testing forward pass with dummy tensors...")

    batch_size = 32
    seq_1m = torch.randn(batch_size, 60, 36) # 60 candles of 1m
    seq_5m = torch.randn(batch_size, 12, 36) # 12 candles of 5m (equivalent to 60m)
    regime = torch.randint(0, 8, (batch_size,)) # 8 regimes

    out = model(seq_1m, seq_5m, regime)
    print("Output Shapes:")
    for k, v in out.items():
        print(f" - {k}: {v.shape}")
    print("✅ Forward pass successful.")
