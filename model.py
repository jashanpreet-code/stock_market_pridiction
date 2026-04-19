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


data = yf.download('AAPL', start='2010-01-01', end=dt.datetime.now().strftime('%Y-%m-%d'))
data.to_csv('apple.csv')
print(data.head())
print(data.shape)
print(data.info())