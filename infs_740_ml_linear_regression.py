import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error

def predict_next_closing_price_symbol():
  # Fetch stock data
  ticker_symbol = 'AAPL'  ##use any symbol
  ticker_symbol = 'GOOGL'
  data = yf.download(ticker_symbol, start='2020-01-01', end='2025-01-01')

  # Prepare data, ensure it remains as DataFrame
  data = pd.DataFrame(data['Close'])
  data['Prev Close'] = data['Close'].shift(1)
  data.dropna(inplace=True)  # Remove any rows with NaN values
  # Setup features and target
  X = data[['Prev Close']]
  y = data['Close']
  data.sort_index(inplace=True)

  # Split the data into train and test sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  # Model training using LinearRegression
  model = LinearRegression()
  model.fit(X_train, y_train)
  # Predicting the stock prices
  predictions = model.predict(X_test)
  predictions = pd.Series(predictions, index=X_test.index)  # Convert predictions to a pandas Series with proper index
  # Calculate the RMSE
  '''
  The root-mean-square deviation (RMSD) or root-mean-square error (RMSE)
  is one of frequently used measures of the differences between true or predicted values on the one hand and observed values or an estimator on the other.
  '''
  rmse = np.sqrt(mean_squared_error(y_test, predictions))

  print("Root Mean Squared Error:", rmse)
  # Create a DataFrame for plotting to handle indices seamlessly
  results = pd.DataFrame({
    'Actual': y_test,
    'Predicted': predictions
  })
  # Plotting using Seaborn and Matplotlib
  plt.figure(figsize=(10, 6))
  sns.lineplot(data=results)
  plt.title(f'Actual vs Predicted Stock Prices for {ticker_symbol}')
  plt.xlabel('Date')
  plt.ylabel('Stock Price')
  plt.legend(title='Legend', labels=['Actual', 'Predicted'])
  plt.show()

if __name__=='__main__':
  predict_next_closing_price_symbol()
