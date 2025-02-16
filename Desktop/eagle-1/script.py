import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np

# 1. Download historical Bitcoin data (last 10 years for better accuracy)
btc = yf.download("BTC-USD", period="10y", interval="1d")

# 2. Create additional columns to improve the forecast
btc["Volume"] = btc["Volume"].rolling(window=7, min_periods=1).mean()
btc["Moving_Avg"] = btc["Close"].rolling(window=50, min_periods=1).mean()

# Technical indicators
btc['RSI'] = 100 - (100 / (1 + btc['Close'].pct_change().rolling(21).mean()))
short_ema = btc['Close'].ewm(span=12, adjust=False).mean()
long_ema = btc['Close'].ewm(span=26, adjust=False).mean()
btc['MACD'] = short_ema - long_ema
btc['Bollinger_Upper'] = btc['Close'].rolling(window=20).mean() + (btc['Close'].rolling(window=20).std() * 2)
btc['Bollinger_Lower'] = btc['Close'].rolling(window=20).mean() - (btc['Close'].rolling(window=20).std() * 2)

btc.fillna(method="bfill", inplace=True)

# 3. Fetch sentiment data using VADER
def fetch_sentiment():
    analyzer = SentimentIntensityAnalyzer()
    headlines = ["Bitcoin is rising!", "Market is bearish on BTC", "BTC hits all-time high!"]  # Replace with live data
    scores = [analyzer.polarity_scores(h)['compound'] for h in headlines]
    return np.mean(scores)

btc['Sentiment'] = fetch_sentiment()

# 4. Prepare data for Prophet
df = btc.reset_index()[["Date", "Close", "Volume", "Moving_Avg", "RSI", "MACD", "Sentiment"]]
df.columns = ["ds", "y", "volume", "moving_avg", "rsi", "macd", "sentiment"]

# 5. Create and configure the Prophet model
model = Prophet(daily_seasonality=True)
model.add_regressor("volume")
model.add_regressor("moving_avg")
model.add_regressor("rsi")
model.add_regressor("macd")
model.add_regressor("sentiment")
model.fit(df)

# 6. Create future dates to predict (next 30 days)
future = model.make_future_dataframe(periods=30)
future["volume"] = df["volume"].iloc[-1]
future["moving_avg"] = df["moving_avg"].iloc[-1]
future["rsi"] = df["rsi"].iloc[-1]
future["macd"] = df["macd"].iloc[-1]
future["sentiment"] = fetch_sentiment()

# 7. Make prediction
forecast = model.predict(future)

# 8. Get the current exchange rate from USD to BRL
url = "https://api.exchangerate-api.com/v4/latest/USD"
response = requests.get(url).json()
usd_to_brl = response["rates"]["BRL"]
forecast["BTC_BRL"] = forecast["yhat"] * usd_to_brl

# 9. Display forecast
forecast_30d = forecast[["ds", "yhat", "BTC_BRL"]].tail(30)
forecast_30d.columns = ["Date", "BTC Price (USD)", "BTC Price (BRL)"]
forecast_30d["BTC Price (USD)"] = forecast_30d["BTC Price (USD)"].apply(lambda x: f"{x:,.2f}")
forecast_30d["BTC Price (BRL)"] = forecast_30d["BTC Price (BRL)"].apply(lambda x: f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
print("\nBitcoin price forecast for the next 30 days:")
print(forecast_30d.to_string(index=False))

# 10. Plot results
plt.figure(figsize=(12,6))
plt.plot(df["ds"], df["y"], label="Historical BTC/USD", linewidth=2)
plt.plot(forecast["ds"], forecast["yhat"], label="Forecast BTC/USD", linestyle="dashed", linewidth=2)
plt.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], alpha=0.2, label="Uncertainty")
plt.xlabel("Date")
plt.ylabel("BTC Price (USD)")
plt.title("Accurate Bitcoin Price Forecast with Sentiment & Indicators")
plt.legend()
plt.grid()
plt.show()
