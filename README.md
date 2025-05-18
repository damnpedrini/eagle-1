# EAGLE-1

**EAGLE-1** is an experimental Bitcoin price forecasting system with a 30-day horizon. Built in Python, it combines time series modeling with technical indicators and sentiment analysis to generate more robust predictions. The system leverages Facebook Prophet with custom external regressors.

## Overview

The pipeline includes the following steps:

- Fetching historical BTC-USD price data from public APIs
- Calculating technical indicators (MA50, RSI, MACD, Bollinger Bands)
- Performing market sentiment analysis using VADER (NLTK)
- Converting USD-based predictions into BRL using exchange rate APIs
- Training and forecasting with Prophet using additional regressors

## Features

- **Time Series Modeling**: Based on Prophet with daily seasonality
- **Technical Indicators**: Integrated as regressors to improve forecast accuracy
- **Sentiment Input**: Real-time or static news sentiment scores influence projections
- **Currency Conversion**: Final output includes prices in both USD and BRL

## Requirements

- Python 3.9+
- `pandas`
- `yfinance`
- `requests`
- `prophet`
- `nltk`

To install dependencies:

```bash
pip install pandas yfinance requests prophet nltk
```

## Output

The model outputs a 30-day forecast for the price of Bitcoin in both USD and BRL.

Example:

```
           ds       yhat    BTC_BRL
2025-06-01  72134.12   371114.08
2025-06-02  72301.45   372012.33
...
```

## Notes

- This project is for **experimental and educational purposes only**.
- Forecast accuracy depends on data quality and feature selection.
- For production use, additional steps such as backtesting, feature scaling, and automated news ingestion are recommended.

---
