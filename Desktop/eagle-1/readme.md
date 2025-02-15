
# Eagle-1 Model Price Prediction (Bitcoin) 🦅

This project is designed to forecast cryptocurrency prices, with a focus on **memecoins** like **Bitcoin**. The goal is to utilize historical price data and machine learning techniques to make predictions for the next 7 days.

## Features

- **Price Prediction**: The model forecasts Bitcoin prices for the next 7 days based on the latest historical data.
- **Moving Averages**: Implements 14-day moving averages and rolling volume data to improve prediction accuracy.
- **BRL Conversion**: Converts the predicted Bitcoin price from USD to BRL based on the current exchange rate.
- **Visualization**: Displays a graph showing the predictions along with uncertainty intervals.

## Dependencies

The project relies on the following Python libraries:

- **`yfinance`**: Used to download historical Bitcoin price data.
- **`pandas`**: For data manipulation.
- **`matplotlib`**: To generate visualizations of the data and predictions.
- **`prophet`**: For time series modeling and making predictions.
- **`requests`**: To fetch the current exchange rate from USD to BRL.

To install the required dependencies, you can run:

```bash
pip install yfinance pandas matplotlib prophet requests
```

## How to Use

1. **Download Data**: The project pulls 10 years of historical Bitcoin data from Yahoo Finance.
2. **Prepare the Model**: The script processes the data, adding a 14-day moving average and a rolling volume to enhance prediction accuracy.
3. **Train the Prophet Model**: The Prophet model is trained with the historical data and additional regressors for better forecasting.
4. **Make Predictions**: The model forecasts Bitcoin prices for the next 7 days.
5. **Convert to BRL**: The forecasted Bitcoin price in USD is converted to BRL using the latest exchange rate.
6. **View Results**: The forecast is displayed both in the terminal and as a graph.

## Example Output

When the code is run, you'll see the following output showing the predicted Bitcoin prices for the next 7 days, both in USD and BRL:

```
Bitcoin Price Prediction for the Next 7 Days:
 Date         BTC Price (USD)   BTC Price (BRL)
 2025-02-16    50000             250000
 2025-02-17    50500             252500
 2025-02-18    51000             255000
 ...
```

## Graph

The project also generates a graph that shows:

- The historical Bitcoin price data.
- The predicted Bitcoin price for the next 7 days.
- Uncertainty intervals around the prediction to visualize the model's confidence.

## Conclusion

This project serves as a foundational framework for predicting memecoin prices, with an emphasis on Bitcoin, using machine learning techniques. It can be expanded to include other cryptocurrencies or more advanced prediction models.

## License

The project is licensed under the [MIT License](LICENSE).
