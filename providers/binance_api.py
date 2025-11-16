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

    import pandas as pd
    ohlc = {
        "open": [float(c[1]) for c in data],
        "high": [float(c[2]) for c in data],
        "low": [float(c[3]) for c in data],
        "close": [float(c[4]) for c in data],
        "date": [pd.to_datetime(c[0], unit="ms") for c in data]
    }
    df = pd.DataFrame(ohlc)
    df.set_index("date", inplace=True)
    return df
