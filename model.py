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



# data = yf.download('AAPL', start='2010-01-01', end=dt.datetime.now().strftime('%Y-%m-%d'))
# data.reset_index().to_csv('apple.csv')


# data cleaning and preprocessing
def data_cleaning():
    df = pd.read_csv('apple.csv')
    print(df.columns)
    df = df.drop(columns=['Dividends', 'Stock Splits'])
    df.to_csv('apple.csv', index=False)
    print(df.columns)
    print(df.head())
    print(df.shape)
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    df = df.set_index('Date')
    df.to_csv('apple.csv')
    
def feature_scaling():
    df = pd.read_csv('apple.csv')
    closing_price = df[['Close']].values  
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(closing_price)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    return scaled_data  
    
    
    
    
    
# data_cleaning()
feature_scaling()





# print(ticker.info)

# print(data)
# print(data.head())
# print(data.shape) 
# print(data.info())