import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
import pickle
import os

from data_processing import compute_universal_features, align_timeframes

def train_regime_model(df_1m, df_5m, n_components=8, save_path="Universal_Trading_Models/hmm_regime_model.pkl"):
    """
    Trains an 8-state Gaussian Hidden Markov Model (HMM) on robust MACRO features
    to detect market regimes.
    """
    print(f"[{pd.Timestamp.now()}] Preparing features for HMM training...")

    df_1m, feats_1m = compute_universal_features(df_1m.copy())
    df_5m, feats_5m = compute_universal_features(df_5m.copy())

    merged_df = align_timeframes(df_1m, df_5m)
    merged_df.dropna(inplace=True)

    # We select a core subset of exactly 8 macro features for the HMM
    hmm_features = [
        'Log_Return', 'NATR_14', 'Dist_EMA_50', 'Hist_Vol_20',
        'Log_Return_5m', 'NATR_50_5m', 'Dist_EMA_200_5m', 'RSI_14_5m'
    ]

    X = merged_df[hmm_features].values

    print(f"[{pd.Timestamp.now()}] Training HMM with {n_components} states on {X.shape[0]} samples...")

    # Use diagonal covariance for extreme stability even on smaller datasets.
    model = GaussianHMM(
        n_components=n_components,
        covariance_type="diag",
        n_iter=100,
        random_state=42,
        verbose=True
    )

    # Adding a tiny bit of noise prevents singular covariance errors
    noise = np.random.normal(0, 1e-8, X.shape)
    model.fit(X + noise)

    if model.monitor_.converged:
        print(f"✅ HMM Converged after {model.monitor_.iter} iterations.")
    else:
        print("⚠️ HMM did not converge perfectly. (Consider more data).")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "wb") as f:
        pickle.dump(model, f)

    print(f"💾 HMM Model saved to {save_path}")

    hidden_states = model.predict(X)
    merged_df['Regime'] = hidden_states

    print("\nRegime Distribution (State 0-7):")
    print(merged_df['Regime'].value_counts(normalize=True).sort_index())

    return model, merged_df

if __name__ == "__main__":
    import os
    if os.path.exists('sample_1m.csv'):
        df_1m = pd.read_csv('sample_1m.csv')
        df_5m = pd.read_csv('sample_5m.csv')
        train_regime_model(df_1m, df_5m)
    else:
        print("Skipping direct run because sample data is missing in this directory.")
