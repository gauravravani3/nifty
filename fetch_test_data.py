from SmartApi import SmartConnect
import pyotp
import pandas as pd
from datetime import datetime, timedelta
import time
import os

# === LOGIN SETUP ===
# Securely fetching credentials from environment variables
userid = os.environ.get('ANGEL_USER_ID')
apiKey = os.environ.get('ANGEL_API_KEY')
pin = os.environ.get('ANGEL_PIN')
totpKey = os.environ.get('ANGEL_TOTP_KEY')

if not all([userid, apiKey, pin, totpKey]):
    print("⚠️ Warning: One or more Angel One credentials (ANGEL_USER_ID, ANGEL_API_KEY, ANGEL_PIN, ANGEL_TOTP_KEY) are missing in environment variables.")
    # Exit early if we can't authenticate securely
    exit(1)

obj = SmartConnect(api_key=apiKey)
otp = pyotp.TOTP(totpKey).now()
data = obj.generateSession(userid, pin, otp)
print("✅ Login Success")

# === FUNCTION TO FETCH DATA ===
def fetch_chunk(exchange, symboltoken, interval, start, end, max_retries=3):
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
            if resp and resp.get("status") is True and resp.get("data"):
                return resp["data"]
            time.sleep(1)
        except Exception:
            time.sleep(1)
        attempt += 1
    return []

# Test data fetch strictly before 23/02/2025
end_date = datetime.strptime("2024-02-15 15:30", "%Y-%m-%d %H:%M")
start_date = end_date - timedelta(days=5)

candles_1m = fetch_chunk("NSE", "99926000", "ONE_MINUTE", start_date, end_date)
df_1m = pd.DataFrame(candles_1m, columns=["Datetime", "Open", "High", "Low", "Close", "Volume"])
df_1m.to_csv('/app/sample_1m.csv', index=False)

candles_5m = fetch_chunk("NSE", "99926000", "FIVE_MINUTE", start_date, end_date)
df_5m = pd.DataFrame(candles_5m, columns=["Datetime", "Open", "High", "Low", "Close", "Volume"])
df_5m.to_csv('/app/sample_5m.csv', index=False)

print("Data fetched and saved to /app/")
obj.terminateSession(userid)
