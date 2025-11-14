import requests

def get_historical(symbol="BTCUSDT", interval="1d", limit=200):
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    closes = [float(c[4]) for c in data]
    return closes
