import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Import our custom modules
from data_processing import compute_universal_features, align_timeframes, calculate_targets
from train_hmm import train_regime_model
from model_architecture import UniversalTradingTransformer

class TradingDataset(Dataset):
    def __init__(self, df, seq_len_1m=60, seq_len_5m=12):
        self.df = df
        self.seq_len_1m = seq_len_1m
        self.seq_len_5m = seq_len_5m

        # 36 Universal Features (from 1m and 5m columns)
        self.features_1m = [
            'Log_Return', 'Body_Ratio', 'Upper_Wick_Ratio', 'Lower_Wick_Ratio', 'High_Low_Spread',
            'Dist_EMA_9', 'Dist_EMA_21', 'Dist_EMA_50', 'Dist_EMA_200',
            'Norm_MACD_Line', 'Norm_MACD_Signal', 'Norm_MACD_Hist',
            'ADX', 'Plus_DI', 'Minus_DI', 'Aroon_Up', 'Aroon_Down',
            'RSI_14', 'Stoch_K', 'Stoch_D', 'Stoch_RSI_K', 'CCI', 'WPR', 'ROC_9', 'ROC_21', 'TSI',
            'NATR_14', 'NATR_50', 'BB_P_Band', 'BB_Width', 'Donchian_P_Band', 'Donchian_Width',
            'Keltner_P_Band', 'Hist_Vol_20', 'Dist_20_High', 'Dist_20_Low'
        ]

        self.features_5m = [f"{col}_5m" for col in self.features_1m]

        # We drop the first elements to ensure we have enough history to build sequences
        self.valid_indices = self.df.index[max(self.seq_len_1m, self.seq_len_5m):]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        end_idx = self.df.index.get_loc(self.valid_indices[idx])

        # Sequence Windows
        seq_1m_data = self.df.iloc[end_idx - self.seq_len_1m : end_idx][self.features_1m].values
        seq_5m_data = self.df.iloc[end_idx - self.seq_len_5m : end_idx][self.features_5m].values

        # Regime State at current timestep
        regime = self.df.iloc[end_idx]['Regime']

        # Targets at current timestep
        target_dir = self.df.iloc[end_idx]['Target_TP_Hit']
        target_time = self.df.iloc[end_idx]['Target_Time_to_Event']
        target_strength = self.df.iloc[end_idx]['Target_Move_Strength']

        # Dynamic Optimal ATR Calculation (Simple mock heuristic for now based on regime and volatility)
        optimal_atr = 1.5 if target_dir == 1 else 1.2
        entry_delay = 0.0 # Placeholder for pullback optimization

        return {
            'seq_1m': torch.tensor(seq_1m_data, dtype=torch.float32),
            'seq_5m': torch.tensor(seq_5m_data, dtype=torch.float32),
            'regime': torch.tensor(int(regime), dtype=torch.long),
            't_dir': torch.tensor([target_dir], dtype=torch.float32),
            't_time': torch.tensor([target_time], dtype=torch.float32),
            't_delay': torch.tensor([entry_delay], dtype=torch.float32),
            't_atr': torch.tensor([optimal_atr], dtype=torch.float32),
            't_strength': torch.tensor([target_strength], dtype=torch.float32)
        }

def train_transformer(df, epochs=5, batch_size=64, save_dir="Universal_Trading_Models"):
    print(f"\n[{pd.Timestamp.now()}] Initializing Multi-Head Transformer Training...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using Device: {device}")

    # Check if GPU is present for Google Colab
    if device.type == 'cuda':
        print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(0)/(1024**2):.2f} MB")

    dataset = TradingDataset(df)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = UniversalTradingTransformer().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # Loss functions for the 6 Heads
    bce_loss = nn.BCELoss() # For Direction, Confidence, Move Strength
    mse_loss = nn.MSELoss() # For Time, Delay, ATR

    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, 'universal_transformer.pth')

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for batch in progress_bar:
            seq_1m = batch['seq_1m'].to(device)
            seq_5m = batch['seq_5m'].to(device)
            regime = batch['regime'].to(device)

            # Targets
            t_dir = batch['t_dir'].to(device)
            t_time = batch['t_time'].to(device)
            t_delay = batch['t_delay'].to(device)
            t_atr = batch['t_atr'].to(device)
            t_strength = batch['t_strength'].to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(seq_1m, seq_5m, regime)

            # Calculate Losses
            loss_dir = bce_loss(outputs['direction'], t_dir)
            loss_time = mse_loss(outputs['time_to_event'], t_time)
            loss_delay = mse_loss(outputs['entry_delay'], t_delay)
            loss_atr = mse_loss(outputs['atr_mult'], t_atr)

            # Confidence is trained to match the actual direction probability
            loss_conf = bce_loss(outputs['confidence'], t_dir)
            loss_strength = bce_loss(outputs['move_strength'], t_strength)

            # Weighted Total Loss (Weights can be tuned)
            loss = (loss_dir * 1.0) + (loss_time * 0.1) + (loss_delay * 0.1) + \
                   (loss_atr * 0.1) + (loss_conf * 0.5) + (loss_strength * 0.5)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'Loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f} | Time: {time.time()-start_time:.1f}s")

    torch.save(model.state_dict(), model_path)
    print(f"💾 Transformer Model saved to {model_path}")

def run_full_pipeline():
    print("==================================================")
    print("🚀 UNIVERSAL TRADING MODEL: FULL TRAINING PIPELINE")
    print("==================================================")

    if not os.path.exists('sample_1m.csv') or not os.path.exists('sample_5m.csv'):
        print("❌ Error: sample_1m.csv and sample_5m.csv not found.")
        print("Please fetch data first using the Angel One API script.")
        return

    df_1m = pd.read_csv('sample_1m.csv')
    df_5m = pd.read_csv('sample_5m.csv')

    # 1. Train HMM Regime Model (Returns Dataframe with 'Regime' column appended)
    print("\n--- STEP 1: TRAINING HMM REGIME MODEL ---")
    hmm_model, merged_df = train_regime_model(df_1m, df_5m)

    # 2. Calculate Complex Targets (TP/SL, Time, Strength) on merged dataset
    print("\n--- STEP 2: CALCULATING INTRADAY TARGETS ---")
    print(f"Dataset Size Before Targets: {merged_df.shape}")
    merged_df = calculate_targets(merged_df, sl_atr_multiplier=1.5, tp_atr_multiplier=3.0, max_candles_ahead=15)
    print("Targets Computed. Example distribution of Direction (TP=1, SL=0):")
    print(merged_df['Target_TP_Hit'].value_counts())

    # Replace NaNs that might have trickled in during target calculation or shifting
    merged_df.fillna(0, inplace=True)

    # 3. Train Transformer
    print("\n--- STEP 3: TRAINING MULTI-HEAD TRANSFORMER ---")
    # Using 5 epochs for testing purposes. In Colab, this can be increased to 50-100.
    train_transformer(merged_df, epochs=5, batch_size=32)

    print("\n✅ FULL PIPELINE EXECUTED SUCCESSFULLY.")
    print("Model files are ready in 'Universal_Trading_Models' folder.")

if __name__ == "__main__":
    run_full_pipeline()
