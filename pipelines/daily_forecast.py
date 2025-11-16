from providers.binance_api import get_historical
from core.forecasting import moving_average
import mplfinance as mpf

def run_daily_forecast(symbol="BTC", days=30):
    pair = f"{symbol.upper()}USDT"

    df = get_historical(pair)

    print(f"\nCandlestick chart for {symbol} (last 7 days):")
    last7 = df.tail(7)
    mpf.plot(last7, type='candle', style='nightclouds', title='Eagle-1 BTC ESTIMATE USD', ylabel='Price (USD)', volume=False)
