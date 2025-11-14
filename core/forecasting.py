def moving_average(series, window=7):
    if len(series) < window:
        raise ValueError("Not Enaught data to calculate moving average.")
    return sum(series[-window:]) / window