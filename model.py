import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input
import pickle

# 1. Setup Variables
# Swapped TATAMOTORS.NS for RELIANCE.NS to ensure reliable data
tickers = ['AAPL', 'MSFT', 'TSLA', 'RELIANCE.NS', 'NVDA']

X_train_master, y_train_master = [], []
X_test_master, y_test_master = [], []

scalers_dict = {}
lookback = 60

# 2. Processing Loop for all Stocks
print("Starting data processing...")

for ticker in tickers:
    print(f"Fetching and formatting data for {ticker}...")
    
    # Download 10 years of live data 
    df = yf.Ticker(ticker).history(period='10y')
    
    # --- SAFETY NET FOR API FAILURES ---
    if df.empty:
        print(f"⚠️ WARNING: No data found for {ticker}. Skipping to next stock...")
        continue
    # -----------------------------------
    
    # Clean Data (Isolate 'Close' price and drop missing days)
    df = df[['Close']].dropna()
    
    # Scale Data between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df.values)
    
    # Save the specific math rule for this stock into our dictionary
    scalers_dict[ticker] = scaler 
    
    # Create Sliding Windows (Flashcards)
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
        
    X, y = np.array(X), np.array(y)
    # Reshape to 3D for the AI (Samples, Time Steps, Features)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # Split Data (80% Train, 20% Test)
    split_index = int(len(X) * 0.8)
    
    # Dump the formatted data into the Master Buckets
    X_train_master.extend(X[:split_index])
    y_train_master.extend(y[:split_index])
    X_test_master.extend(X[split_index:])
    y_test_master.extend(y[split_index:])

# 3. Finalize Data
# Save the dictionary of all scalers for later use
with open('scalers_dict.pkl', 'wb') as f:
    pickle.dump(scalers_dict, f)

# Convert the giant Master Lists into final Numpy Arrays
X_train = np.array(X_train_master)
y_train = np.array(y_train_master)
X_test = np.array(X_test_master)
y_test = np.array(y_test_master)

print("\nData processing complete! All valid stocks are ready.")
print(f"Total training samples: {X_train.shape[0]}")
print(f"Total testing samples: {X_test.shape[0]}")

# 4. Build the Global LSTM Model
print("\nBuilding the LSTM Architecture...")
model = Sequential()
model.add(Input(shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=True))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
print("Model Built Successfully! Ready for Phase 2.")