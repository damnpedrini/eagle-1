import yfinance as yf
import pandas as pd
import requests
from prophet import Prophet
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from pandas_datareader import data as pdr


btc = yf.download("BTC-USD", period="1y", interval="1d", auto_adjust=True)
if btc.empty:
    raise RuntimeError("Sem dados: verifique conexão ou símbolo.")


da_close = btc["Close"]
ma50 = da_close.rolling(50, min_periods=1).mean()
ma200 = da_close.rolling(200, min_periods=1).mean()
btc["MA50"] = ma50
btc["MA200"] = ma200

#
btc["Mayer_Multiple"] = da_close / ma200


delta = da_close.diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(14, min_periods=14).mean()
avg_loss = loss.rolling(14, min_periods=14).mean()
btc["RSI"] = 100 - (100 / (1 + avg_gain / avg_loss))


short_ema = da_close.ewm(span=12, adjust=False).mean()
long_ema = da_close.ewm(span=26, adjust=False).mean()
btc["MACD"] = short_ema - long_ema


bb_mean = da_close.rolling(20).mean()
bb_std = da_close.rolling(20).std()
btc["BB_Upper"] = bb_mean + 2 * bb_std
btc["BB_Lower"] = bb_mean - 2 * bb_std
btc["BB_Width"] = btc["BB_Upper"] - btc["BB_Lower"]


btc = btc.bfill()


nltk.download("vader_lexicon")
analyzer = SentimentIntensityAnalyzer()
def fetch_sentiment(headlines: list[str]) -> float:
    scores = [analyzer.polarity_scores(h)["compound"] for h in headlines]
    return sum(scores) / len(scores) if scores else 0
btc["Sentiment"] = fetch_sentiment([
    "Bitcoin explodindo alta!",
    "Mercado bearish em BTC",
    "BTC atinge novo recorde!"
])


resp = requests.get("https://open.er-api.com/v6/latest/USD")
if resp.status_code != 200:
    raise RuntimeError("Falha ao obter taxa USD→BRL")
usd_to_brl = resp.json()["rates"]["BRL"]


start_date = btc.index.min().date()
m2 = pdr.DataReader('M2SL', 'fred', start=start_date)
m2.rename(columns={'M2SL': 'M2_Dollar'}, inplace=True)
daily_m2 = m2.resample('D').ffill().reindex(btc.index)
btc['M2_Dollar'] = daily_m2['M2_Dollar']
btc['M2_Dollar'] = btc['M2_Dollar'].bfill().ffill()


df = btc.reset_index()[["Date", "Close", "MA50", "RSI", "MACD", "BB_Width", "Sentiment", "Mayer_Multiple", "M2_Dollar"]]
df.columns = ["ds", "y", "ma50", "rsi", "macd", "bb_width", "sentiment", "mayer_multiple", "m2_dollar"]
df["ds"] = pd.to_datetime(df["ds"])
df.dropna(inplace=True)

model = Prophet(daily_seasonality=True)
for reg in ["ma50", "rsi", "macd", "bb_width", "sentiment", "mayer_multiple", "m2_dollar"]:
    model.add_regressor(reg)
model.fit(df)

horizon = 30
future = model.make_future_dataframe(periods=horizon, freq='D')
last = df.iloc[-1]
for reg in ["ma50", "rsi", "macd", "bb_width", "sentiment", "mayer_multiple", "m2_dollar"]:
    future[reg] = last[reg]

forecast = model.predict(future)
forecast["BTC_BRL"] = forecast["yhat"] * usd_to_brl

print(forecast[["ds", "yhat", "BTC_BRL"]].tail(horizon).to_string(index=False))