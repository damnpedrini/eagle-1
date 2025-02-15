import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import requests

# 1. Download historical Bitcoin data (last 10 years for better accuracy)
btc = yf.download("BTC-USD", period="10y", interval="1d")

# 2. Create additional columns to improve the forecast
btc["Volume"] = btc["Volume"].rolling(window=7, min_periods=1).mean()  # Moving average of volume
btc["Moving_Avg"] = btc["Close"].rolling(window=14, min_periods=1).mean()  # 14-day moving average

# Fill NaN values correctly
btc.fillna(method="bfill", inplace=True)

# 3. Prepare data for Prophet
df = btc.reset_index()[["Date", "Close", "Volume", "Moving_Avg"]]
df.columns = ["ds", "y", "volume", "moving_avg"]

# Check for sufficient data
if df.isnull().sum().sum() > 0:
    raise ValueError("There are still NaNs in the DataFrame! Check the data before training.")
if len(df) < 50:
    raise ValueError("Insufficient data available! The period may be too short.")

# 4. Create and configure the Prophet model with additional variables
model = Prophet(daily_seasonality=True)
model.add_regressor("volume")
model.add_regressor("moving_avg")
model.fit(df)

# 5. Create future dates to predict (next 7 days)
future = model.make_future_dataframe(periods=7)

# Fill future values with the last known values
future["volume"] = df["volume"].iloc[-1]
future["moving_avg"] = df["moving_avg"].iloc[-1]

# 6. Make prediction
forecast = model.predict(future)

# 7. Get the current exchange rate from USD to BRL
url = "https://api.exchangerate-api.com/v4/latest/USD"
response = requests.get(url).json()
usd_to_brl = response["rates"]["BRL"]

# 8. Add conversion to BRL in the forecast
forecast["BTC_BRL"] = forecast["yhat"] * usd_to_brl

# 9. Display forecast for the next 7 days
forecast_7d = forecast[["ds", "yhat", "BTC_BRL"]].tail(7)
forecast_7d.columns = ["Date", "BTC Price (USD)", "BTC Price (BRL)"]

# Format values for better visualization
forecast_7d["BTC Price (USD)"] = forecast_7d["BTC Price (USD)"].apply(lambda x: f"{x:,.2f}")
forecast_7d["BTC Price (BRL)"] = forecast_7d["BTC Price (BRL)"].apply(lambda x: f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

# Show forecast in terminal
print("\nBitcoin price forecast for the next 7 days:")
print(forecast_7d.to_string(index=False))

# 10. Plot the results
plt.figure(figsize=(12,6))
plt.plot(df["ds"], df["y"], label="Historical BTC/USD", linewidth=2)
plt.plot(forecast["ds"], forecast["yhat"], label="Forecast BTC/USD", linestyle="dashed", linewidth=2)
plt.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], alpha=0.2, label="Uncertainty")
plt.xlabel("Date")
plt.ylabel("BTC Price (USD)")
plt.title("Accurate Bitcoin Price Forecast (Next 7 Days)")
plt.legend()
plt.grid()
plt.show()
