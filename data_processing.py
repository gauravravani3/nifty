import pandas as pd
import numpy as np
import ta

def compute_universal_features(df):
    """
    Computes 36 universal, scale-invariant features.
    Volume is strictly ignored because indices like Nifty 50 have 0 volume.
    """
    if 'Datetime' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df.set_index('Datetime', inplace=True)

    # 1. Price Action (Micro-Structure)
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

    # Intraday mask: set Log_Return to 0 at 09:15 to remove overnight gap effect
    df.loc[df.index.time == pd.to_datetime('09:15').time(), 'Log_Return'] = 0.0

    true_range = np.maximum(df['High'] - df['Low'], 1e-8)
    df['Body_Ratio'] = (df['Close'] - df['Open']) / true_range
    df['Upper_Wick_Ratio'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / true_range
    df['Lower_Wick_Ratio'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / true_range
    df['High_Low_Spread'] = (df['High'] - df['Low']) / df['Close'].shift(1).replace(0, np.nan)

    # 2. Trend & Moving Averages (% Distance from Price)
    df['EMA_9'] = ta.trend.ema_indicator(df['Close'], window=9)
    df['Dist_EMA_9'] = (df['Close'] - df['EMA_9']) / df['EMA_9']

    df['EMA_21'] = ta.trend.ema_indicator(df['Close'], window=21)
    df['Dist_EMA_21'] = (df['Close'] - df['EMA_21']) / df['EMA_21']

    df['EMA_50'] = ta.trend.ema_indicator(df['Close'], window=50)
    df['Dist_EMA_50'] = (df['Close'] - df['EMA_50']) / df['EMA_50']

    df['EMA_200'] = ta.trend.ema_indicator(df['Close'], window=200)
    df['Dist_EMA_200'] = (df['Close'] - df['EMA_200']) / df['EMA_200']

    macd = ta.trend.MACD(df['Close'])
    df['Norm_MACD_Line'] = macd.macd() / df['Close']
    df['Norm_MACD_Signal'] = macd.macd_signal() / df['Close']
    df['Norm_MACD_Hist'] = macd.macd_diff() / df['Close']

    adx_indicator = ta.trend.ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'])
    df['ADX'] = adx_indicator.adx()
    df['Plus_DI'] = adx_indicator.adx_pos()
    df['Minus_DI'] = adx_indicator.adx_neg()

    aroon = ta.trend.AroonIndicator(high=df['High'], low=df['Low'])
    df['Aroon_Up'] = aroon.aroon_up()
    df['Aroon_Down'] = aroon.aroon_down()

    # 3. Momentum & Oscillators
    df['RSI_14'] = ta.momentum.rsi(df['Close'], window=14)
    stoch = ta.momentum.StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'])
    df['Stoch_K'] = stoch.stoch()
    df['Stoch_D'] = stoch.stoch_signal()

    stoch_rsi = ta.momentum.StochRSIIndicator(close=df['Close'])
    df['Stoch_RSI_K'] = stoch_rsi.stochrsi_k()

    df['CCI'] = ta.trend.cci(high=df['High'], low=df['Low'], close=df['Close'])
    df['WPR'] = ta.momentum.williams_r(high=df['High'], low=df['Low'], close=df['Close'])
    df['ROC_9'] = ta.momentum.roc(df['Close'], window=9)
    df['ROC_21'] = ta.momentum.roc(df['Close'], window=21)
    df['TSI'] = ta.momentum.tsi(df['Close'])

    # 4. Volatility & Channels (Normalized)
    atr_14 = ta.volatility.average_true_range(high=df['High'], low=df['Low'], close=df['Close'], window=14)
    df['NATR_14'] = atr_14 / df['Close']

    atr_50 = ta.volatility.average_true_range(high=df['High'], low=df['Low'], close=df['Close'], window=50)
    df['NATR_50'] = atr_50 / df['Close']

    bb = ta.volatility.BollingerBands(close=df['Close'])
    df['BB_P_Band'] = bb.bollinger_pband()
    df['BB_Width'] = bb.bollinger_wband()

    donchian = ta.volatility.DonchianChannel(high=df['High'], low=df['Low'], close=df['Close'])
    df['Donchian_P_Band'] = donchian.donchian_channel_pband()
    df['Donchian_Width'] = donchian.donchian_channel_wband()

    keltner = ta.volatility.KeltnerChannel(high=df['High'], low=df['Low'], close=df['Close'])
    df['Keltner_P_Band'] = keltner.keltner_channel_pband()

    df['Hist_Vol_20'] = df['Log_Return'].rolling(window=20).std()

    rolling_high = df['High'].rolling(20).max()
    rolling_low = df['Low'].rolling(20).min()
    df['Dist_20_High'] = (df['Close'] - rolling_high) / rolling_high
    df['Dist_20_Low'] = (df['Close'] - rolling_low) / rolling_low

    df.fillna(0, inplace=True)

    features = [
        'Log_Return', 'Body_Ratio', 'Upper_Wick_Ratio', 'Lower_Wick_Ratio', 'High_Low_Spread',
        'Dist_EMA_9', 'Dist_EMA_21', 'Dist_EMA_50', 'Dist_EMA_200',
        'Norm_MACD_Line', 'Norm_MACD_Signal', 'Norm_MACD_Hist',
        'ADX', 'Plus_DI', 'Minus_DI', 'Aroon_Up', 'Aroon_Down',
        'RSI_14', 'Stoch_K', 'Stoch_D', 'Stoch_RSI_K', 'CCI', 'WPR', 'ROC_9', 'ROC_21', 'TSI',
        'NATR_14', 'NATR_50', 'BB_P_Band', 'BB_Width', 'Donchian_P_Band', 'Donchian_Width',
        'Keltner_P_Band', 'Hist_Vol_20', 'Dist_20_High', 'Dist_20_Low'
    ]

    cols_to_drop = ['EMA_9', 'EMA_21', 'EMA_50', 'EMA_200']
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

    return df, features

def align_timeframes(df_1m, df_5m):
    """
    Aligns 5m data to 1m data using forward fill to prevent lookahead bias.
    """
    df_5m_shifted = df_5m.shift(1)
    df_5m_shifted.columns = [f"{c}_5m" for c in df_5m_shifted.columns]

    merged = df_1m.join(df_5m_shifted, how='left')
    merged.ffill(inplace=True)
    merged.fillna(0, inplace=True)

    return merged

def calculate_targets(df_1m, sl_atr_multiplier=1.5, tp_atr_multiplier=3.0, max_candles_ahead=15):
    """
    Calculates dynamic Targets (TP/SL) using ATR within exactly 15 candles.
    Added 'Move_Strength' to define if the hit was clean (Strong) or dirty/whipsaw (Weak).
    """
    datetimes = df_1m.index.to_series()
    close_prices = df_1m['Close'].values
    high_prices = df_1m['High'].values
    low_prices = df_1m['Low'].values
    atrs = df_1m['NATR_14'].values * df_1m['Close'].values

    n = len(df_1m)

    tp_hits = np.zeros(n)
    sl_hits = np.zeros(n)
    time_to_events = np.zeros(n)
    move_strengths = np.zeros(n) # 1 for Strong (Direct hit), 0 for Weak (Slow/Choppy hit)

    end_of_day_time = pd.to_datetime('15:15').time()
    cutoff_time = pd.to_datetime('15:30').time()

    for i in range(n):
        current_time = datetimes.iloc[i].time()

        if current_time >= end_of_day_time:
            continue

        entry_price = close_prices[i]
        atr = atrs[i]

        if atr == 0:
            continue

        tp_price = entry_price + (atr * tp_atr_multiplier)
        sl_price = entry_price - (atr * sl_atr_multiplier)

        tp_hit_idx = -1
        sl_hit_idx = -1
        max_drawdown = 0.0 # To measure how close we came to SL before hitting TP

        for j in range(i + 1, min(i + max_candles_ahead, n)):
            if datetimes.iloc[j].date() != datetimes.iloc[i].date():
                break
            if datetimes.iloc[j].time() >= cutoff_time:
                break

            # Record maximum adverse excursion (drawdown)
            drawdown = entry_price - low_prices[j]
            if drawdown > max_drawdown:
                max_drawdown = drawdown

            if high_prices[j] >= tp_price and tp_hit_idx == -1:
                tp_hit_idx = j

            if low_prices[j] <= sl_price and sl_hit_idx == -1:
                sl_hit_idx = j

            if tp_hit_idx != -1 or sl_hit_idx != -1:
                break

        if tp_hit_idx != -1 and (sl_hit_idx == -1 or tp_hit_idx < sl_hit_idx):
            tp_hits[i] = 1
            sl_hits[i] = 0
            time_to_events[i] = tp_hit_idx - i

            # Strong move: if drawdown was less than half of SL distance, it's a very clean, strong breakout.
            if max_drawdown < (atr * sl_atr_multiplier * 0.5):
                move_strengths[i] = 1
            else:
                move_strengths[i] = 0 # Weak/Choppy hit

        elif sl_hit_idx != -1 and (tp_hit_idx == -1 or sl_hit_idx < tp_hit_idx):
            tp_hits[i] = 0
            sl_hits[i] = 1
            time_to_events[i] = sl_hit_idx - i
            move_strengths[i] = 0 # A loss is always considered weak
        else:
            tp_hits[i] = 0
            sl_hits[i] = 0
            time_to_events[i] = 0
            move_strengths[i] = 0

    df_1m['Target_TP_Hit'] = tp_hits
    df_1m['Target_SL_Hit'] = sl_hits
    df_1m['Target_Time_to_Event'] = time_to_events
    df_1m['Target_Move_Strength'] = move_strengths

    return df_1m
