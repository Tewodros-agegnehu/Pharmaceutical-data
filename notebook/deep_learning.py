# -*- coding: utf-8 -*-
"""deep learning.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1zCmNH075PKwh-pPUoDpZ3TtLNpsbnRB3
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

data = pd.read_csv("/content/drive/MyDrive/week 4/train.csv")
data.head()

#extract data for store 1
store_id = 1
store_data = data[data['Store'] == store_id]
store_data.head()

# Sales vs Customers

from matplotlib import pyplot as plt
store_data.plot(kind='scatter', x='Sales', y='Customers', s=32, alpha=.8)
plt.gca().spines[['top', 'right',]].set_visible(False)

# Sales

from matplotlib import pyplot as plt
store_data['Sales'].plot(kind='hist', bins=20, title='Sales')
plt.gca().spines[['top', 'right',]].set_visible(False)

#extract time series data for store 1
store_data = store_data[['Sales', 'Date']].copy()
store_data['Date'] = pd.to_datetime(store_data['Date'])
store_data.set_index('Date', inplace = True)
store_data.sort_index(inplace= True)
store_data.head()

store_data.isnull().sum()

## check for stationarity

result = adfuller(store_data['Sales'])
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")

def create_supervised_data(series, n_lags=7):
    X, y = [], []
    for i in range(n_lags, len(series)):
        X.append(series[i-n_lags:i])
        y.append(series[i])
    return np.array(X), np.array(y)

n_lags = 7  # Example: Use past 7 days to predict the next day
sales_series = store_data['Sales'].values

# Create inputs (X) and target (y)
X, y = create_supervised_data(sales_series, n_lags)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

scaler_X = MinMaxScaler(feature_range=(-1, 1))
scaler_y = MinMaxScaler(feature_range=(-1, 1))

X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
X_test_scaled = scaler_X.transform(X_test)
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))

model = Sequential([
    LSTM(50, activation='tanh', return_sequences=True, input_shape=(n_lags, 1)),
    LSTM(50, activation='tanh'),
    Dense(1)  # Output layer
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], n_lags, 1))
X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], n_lags, 1))

history = model.fit(
    X_train_scaled, y_train_scaled,
    epochs=20,
    batch_size=32,
    validation_split=0.2
)

y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler.inverse_transform(y_pred_scaled)
y_test = scaler.inverse_transform(y_test_scaled)

from sklearn.metrics import mean_squared_error

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse}")

# Visualize predictions
plt.figure(figsize=(10, 6))
plt.plot(y_test, label="Actual Sales")
plt.plot(y_pred, label="Predicted Sales")
plt.legend()
plt.show()

