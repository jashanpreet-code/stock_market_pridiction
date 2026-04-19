import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input
from sklearn.model_selection import train_test_split
import datetime as dt
import pickle
import tensorflow as tf

# print(tf.__version__)

# data = yf.download('AAPL', start='2010-01-01', end=dt.datetime.now().strftime('%Y-%m-%d'))
# data.reset_index().to_csv('apple.csv')
# data.columns = data.columns.droplevel(1)


ticker = yf.Ticker('AAPL')
ticker.info
# data = ticker.history('max')
# print(data)
# print(data.head())
# print(data.shape)
# print(data.info())