# Eagle-1 Model Price Prediction (Bitcoin)

This project is designed to forecast cryptocurrency prices, with a focus on **memecoins** like **Bitcoin**. The goal is to utilize historical price data and machine learning techniques to make predictions for the next 30 days.

## Version 1.1 - Release Notes

### Changes & Improvements:
- **Extended Forecast Window**: Predictions now cover **30 days** instead of **7 days**.
- **Enhanced Moving Averages**: The **14-day moving average** was replaced with a **50-day moving average** for better trend tracking.
- **Refined RSI Calculation**: The **14-day RSI** has been updated to a **21-day RSI** to provide a more stable overbought/oversold indicator.


## Features

- **Price Prediction**: The model forecasts Bitcoin prices for the next **30 days** based on historical data.
- **Moving Averages**: Implements a **50-day moving average** and rolling volume data to improve prediction accuracy.
- **RSI Calculation**: Now uses **21-day RSI** instead of 14 days for better trend stability.
- **BRL Conversion**: Converts the predicted Bitcoin price from USD to BRL based on the current exchange rate.
- **Visualization**: Displays a graph showing the predictions along with uncertainty intervals.

## Dependencies

The project relies on the following Python libraries:

- **`yfinance`**: Used to download historical Bitcoin price data.
- **`pandas`**: For data manipulation.
- **`matplotlib`**: To generate visualizations of the data and predictions.
- **`prophet`**: For time series modeling and making predictions.
- **`requests`**: To fetch the current exchange rate from USD to BRL.
- **`vaderSentiment`**: To incorporate sentiment analysis into price prediction.

To install the required dependencies, you can run:

```bash
pip install yfinance pandas matplotlib prophet requests vaderSentiment
```

## How to Use

1. **Download Data**: The project pulls 10 years of historical Bitcoin data from Yahoo Finance.
2. **Prepare the Model**: The script processes the data, adding a **50-day moving average** and a rolling volume to enhance prediction accuracy.
3. **Train the Prophet Model**: The Prophet model is trained with the historical data and additional regressors for better forecasting.
4. **Make Predictions**: The model forecasts Bitcoin prices for the next **30 days**.
5. **Convert to BRL**: The forecasted Bitcoin price in USD is converted to BRL using the latest exchange rate.
6. **View Results**: The forecast is displayed both in the terminal and as a graph.

## Prediction Logic (Equation)

The prediction model follows this equation:

### **Bitcoin Price Prediction (BTC/USD)**

\[
P_{t+1} = f(P_t, V_t, MA_t)
\]

Where:
- \( P_t \) = Bitcoin price on day \( t \) (dependent variable)
- \( V_t \) = 7-day rolling average of trading volume (additional regressor)
- \( MA_t \) = 14-day moving average of closing price (additional regressor)
- \( f(\cdot) \) = Prophet model function incorporating seasonality and trends

### **Conversion to BRL**

After predicting the price in USD, the conversion to BRL is done using:

\[
P_{BRL, t+1} = P_{t+1} \times R
\]

Where:
- \( P_{BRL, t+1} \) = Predicted Bitcoin price in BRL
- \( R \) = USD to BRL exchange rate fetched from an external API

## Example Output

When the code is run, you'll see the following output showing the predicted Bitcoin prices for the next **30 days**, both in USD and BRL:

```
Bitcoin Price Prediction for the Next 30 Days:
 Date         BTC Price (USD)   BTC Price (BRL)
 2025-02-16    50000             250000
 2025-02-17    50500             252500
 2025-02-18    51000             255000
 ...
```

## Graph

The project also generates a graph that shows:

- The historical Bitcoin price data.
- The predicted Bitcoin price for the next **30 days**.
- Uncertainty intervals around the prediction to visualize the model's confidence.

## Conclusion

This project serves as a foundational framework for predicting memecoin prices, with an emphasis on Bitcoin, using machine learning techniques. It can be expanded to include other cryptocurrencies, on-chain data, or more advanced prediction models.

## License

The project is licensed under the [MIT License](LICENSE).

