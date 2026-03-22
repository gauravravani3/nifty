import torch
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

from model_architecture import UniversalTradingTransformer
from data_processing import compute_universal_features, align_timeframes

def load_models(model_dir="Universal_Trading_Models"):
    print("Loading models...")

    # Load HMM
    with open(f"{model_dir}/hmm_regime_model.pkl", "rb") as f:
        hmm_model = pickle.load(f)

    # Load Transformer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transformer = UniversalTradingTransformer()
    transformer.load_state_dict(torch.load(f"{model_dir}/universal_transformer.pth", map_location=device))
    transformer.to(device)
    transformer.eval()

    print("Models loaded successfully!")
    return hmm_model, transformer, device

def predict_current_state(df_1m, df_5m, hmm_model, transformer, device):
    """
    Takes the recent historical data, processes it, and outputs the live prediction.
    Ensure df_1m has at least 60 rows, and df_5m has at least 12 rows.
    """
    # 1. Feature Engineering
    df_1m, _ = compute_universal_features(df_1m)
    df_5m, _ = compute_universal_features(df_5m)

    merged_df = align_timeframes(df_1m, df_5m)
    merged_df.dropna(inplace=True)

    # 2. Extract HMM Features for the latest row
    hmm_features = [
        'Log_Return', 'NATR_14', 'Dist_EMA_50', 'Hist_Vol_20',
        'Log_Return_5m', 'NATR_50_5m', 'Dist_EMA_200_5m', 'RSI_14_5m'
    ]

    X_hmm = merged_df[hmm_features].values

    # Predict Regime for the entire sequence (we need it for the transformer)
    regimes = hmm_model.predict(X_hmm)
    merged_df['Regime'] = regimes

    # Get the latest state
    current_regime = int(regimes[-1])

    # 3. Prepare Sequence for Transformer
    # 36 Universal Features (from 1m and 5m columns)
    features_1m = [
        'Log_Return', 'Body_Ratio', 'Upper_Wick_Ratio', 'Lower_Wick_Ratio', 'High_Low_Spread',
        'Dist_EMA_9', 'Dist_EMA_21', 'Dist_EMA_50', 'Dist_EMA_200',
        'Norm_MACD_Line', 'Norm_MACD_Signal', 'Norm_MACD_Hist',
        'ADX', 'Plus_DI', 'Minus_DI', 'Aroon_Up', 'Aroon_Down',
        'RSI_14', 'Stoch_K', 'Stoch_D', 'Stoch_RSI_K', 'CCI', 'WPR', 'ROC_9', 'ROC_21', 'TSI',
        'NATR_14', 'NATR_50', 'BB_P_Band', 'BB_Width', 'Donchian_P_Band', 'Donchian_Width',
        'Keltner_P_Band', 'Hist_Vol_20', 'Dist_20_High', 'Dist_20_Low'
    ]
    features_5m = [f"{col}_5m" for col in features_1m]

    seq_len_1m = 60
    seq_len_5m = 12

    if len(merged_df) < max(seq_len_1m, seq_len_5m):
        return {"error": "Not enough data to form sequences. Wait for more candles."}

    seq_1m_data = merged_df.iloc[-seq_len_1m:][features_1m].values
    seq_5m_data = merged_df.iloc[-seq_len_5m:][features_5m].values

    # Convert to Tensors (Add batch dimension)
    t_seq_1m = torch.tensor(seq_1m_data, dtype=torch.float32).unsqueeze(0).to(device)
    t_seq_5m = torch.tensor(seq_5m_data, dtype=torch.float32).unsqueeze(0).to(device)
    t_regime = torch.tensor([current_regime], dtype=torch.long).to(device)

    # 4. Predict
    with torch.no_grad():
        out = transformer(t_seq_1m, t_seq_5m, t_regime)

    # 5. Format Output
    direction_prob = out['direction'].item()
    confidence = out['confidence'].item()
    time_to_event = int(round(out['time_to_event'].item()))
    entry_delay = int(round(out['entry_delay'].item()))
    atr_mult = round(out['atr_mult'].item(), 2)
    move_strength = out['move_strength'].item()

    signal = "BUY/HOLD" if direction_prob > 0.5 else "SELL/EXIT"
    strength_lbl = "STRONG (Clean Move)" if move_strength > 0.5 else "WEAK (Choppy/Whipsaw)"

    current_close = merged_df.iloc[-1]['Close']
    current_atr = merged_df.iloc[-1]['NATR_14'] * current_close

    # Dynamic TP/SL Levels
    if signal == "BUY/HOLD":
        suggested_sl = current_close - (current_atr * atr_mult)
        suggested_tp = current_close + (current_atr * (atr_mult * 2)) # Assumed 1:2 RR for TP
    else:
        suggested_sl = current_close + (current_atr * atr_mult)
        suggested_tp = current_close - (current_atr * (atr_mult * 2))

    return {
        "Timestamp": merged_df.index[-1].strftime('%Y-%m-%d %H:%M'),
        "Current_Price": round(current_close, 2),
        "Market_Regime_State": current_regime,
        "Signal": signal,
        "Confidence": f"{confidence*100:.1f}%",
        "Move_Quality": strength_lbl,
        "Expected_Candles_to_Hit": max(1, min(15, time_to_event)),
        "Optimal_Entry_Wait_Candles": max(0, min(3, entry_delay)),
        "Dynamic_SL_ATR_Multiplier": atr_mult,
        "Suggested_SL_Price": round(suggested_sl, 2),
        "Suggested_TP_Price": round(suggested_tp, 2)
    }

if __name__ == "__main__":
    print("--- LIVE PREDICTION SCRIPT (TEST MODE) ---")
    try:
        # Load sample data as mock "live" data
        df_1m = pd.read_csv('sample_1m.csv')
        df_5m = pd.read_csv('sample_5m.csv')

        hmm, trans, dev = load_models()

        # Take a slice to mock a "live" moment
        mock_1m = df_1m.copy()
        mock_5m = df_5m.copy()

        res = predict_current_state(mock_1m, mock_5m, hmm, trans, dev)

        print("\n📊 LIVE PREDICTION DASHBOARD:")
        for k, v in res.items():
            print(f"{k}: {v}")

    except Exception as e:
        print(f"Skipping execution (Models or data might not be present in current directory). Error: {e}")
