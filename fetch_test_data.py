from SmartApi import SmartConnect
import pyotp
import pandas as pd
from datetime import datetime, timedelta
import time
import os
import sys

# === LOGIN SETUP ===
# Securely fetching credentials from environment variables
userid = os.environ.get('ANGEL_USER_ID')
apiKey = os.environ.get('ANGEL_API_KEY')
pin = os.environ.get('ANGEL_PIN')
totpKey = os.environ.get('ANGEL_TOTP_KEY')

if not all([userid, apiKey, pin, totpKey]):
    print("⚠️ Warning: One or more Angel One credentials (ANGEL_USER_ID, ANGEL_API_KEY, ANGEL_PIN, ANGEL_TOTP_KEY) are missing in environment variables.")
    # Exit early if we can't authenticate securely
    sys.exit(1)

obj = SmartConnect(api_key=apiKey)
otp = pyotp.TOTP(totpKey).now()
data = obj.generateSession(userid, pin, otp)
print("✅ Login Success")

# === FUNCTION TO FETCH 29 DAYS OF DATA ===
def fetch_chunk(exchange, symboltoken, interval, start, end, max_retries=8, base_sleep=1.0):
    params = {
        "exchange": exchange,
        "symboltoken": symboltoken,
        "interval": interval,
        "fromdate": start.strftime("%Y-%m-%d %H:%M"),
        "todate": end.strftime("%Y-%m-%d %H:%M")
    }

    attempt = 1
    while attempt <= max_retries:
        try:
            resp = obj.getCandleData(params)

            # ✅ STATUS + DATA GUARD
            if resp and resp.get("status") is True and resp.get("data"):
                return resp["data"]

            # ❌ No data / status false → retry SAME chunk
            err = (resp or {}).get("errorcode")
            msg = (resp or {}).get("message")
            print(f"⚠️ No data / API error (retry {attempt}/{max_retries}) | {err} | {msg}")

        except Exception as e:
            print(f"⚠️ Error fetching (retry {attempt}/{max_retries}):", e)

        # backoff sleep (1,2,4,8... capped)
        sleep_s = base_sleep * (2 ** (attempt - 1))
        time.sleep(min(sleep_s, 30))
        attempt += 1

    # If all retries failed, return empty (but loop will NOT advance until you decide)
    return []

# === MULTIPLE CHUNKS LOOP (10 years) ===
def get_historical_long(exchange, symboltoken, interval="ONE_MINUTE", years=10):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years)

    all_data = []
    chunk = timedelta(days=29)
    current_start = start_date
    loop_no = 1

    while current_start < end_date:
        current_end = min(current_start + chunk, end_date)

        # 👇 Print start and end time for each loop
        print(f"🔁 Loop {loop_no}: {current_start.strftime('%Y-%m-%d %H:%M')} → {current_end.strftime('%Y-%m-%d %H:%M')}")

        candles = fetch_chunk(exchange, symboltoken, interval, current_start, current_end)

        # ✅ If after retries still empty → DON'T MOVE AHEAD
        if not candles:
            print("🛑 Chunk failed even after retries. Stopping here (same chunk not advanced).")
            break

        all_data.extend(candles)

        current_start = current_end  # move to next chunk
        loop_no += 1

    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=["Datetime","Open","High","Low","Close","Volume"])
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.drop_duplicates(subset=["Datetime"]).sort_values("Datetime").reset_index(drop=True)
    return df

if __name__ == "__main__":
    print("Fetching 1-MINUTE Data (10 Years)...")
    df_1m = get_historical_long("NSE", "99926000", interval="ONE_MINUTE", years=10)
    print(df_1m.head())
    print(f"📊 Total 1m rows fetched: {len(df_1m)}")
    df_1m.to_csv("sample_1m.csv", index=False)
    print("💾 Saved to sample_1m.csv\n")

    print("Fetching 5-MINUTE Data (10 Years)...")
    df_5m = get_historical_long("NSE", "99926000", interval="FIVE_MINUTE", years=10)
    print(df_5m.head())
    print(f"📊 Total 5m rows fetched: {len(df_5m)}")
    df_5m.to_csv("sample_5m.csv", index=False)
    print("💾 Saved to sample_5m.csv\n")

    obj.terminateSession(userid)
