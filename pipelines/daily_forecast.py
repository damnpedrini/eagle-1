from providers.binance_api import get_historical
from core.forecasting import moving_average

PYTHONPATH=. python app.py

def run_daily_forecast(symbol="BTC", days=30):
    pair = f"{symbol.upper()}USDT"

    prices = get_historical(pair)

    forecast = moving_average(prices, window=7)

    print(f"\n📈 Forecast for {symbol} ({days} days): {forecast}\n")
