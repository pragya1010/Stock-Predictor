from keras import backend as K


import os
from importlib import reload

import numpy as np
import pandas as pd
from keras.layers.core import Dense, Activation, Dropout
#from keras.layers.recurrent import LSTM

from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import math
from keras.layers.core import Dense, Activation, Dropout
import  time
from keras.layers.recurrent import LSTM
from keras.models import Sequential



#Read dataset

import datetime
import pandas_datareader.data as pdb

start = datetime.datetime(2018, 4, 1)
end = datetime.datetime(2018, 10, 24)


df = pdb.DataReader('AAPL', 'yahoo', start, end)
#print(df.columns)
#print(df.info())
df = df[['High', 'Low', 'Open', 'Close', 'Volume']]
apple_close = df['Close']
apple_close = apple_close.values.reshape(len(apple_close), 1)
# plt.plot(apple_close)
# plt.show()

#normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
apple_close = scaler.fit_transform(apple_close)

train_size =  int(len(apple_close)*0.8)
test_size = len(apple_close) - train_size

apple_train = apple_close[0:train_size,:]
apple_test = apple_close[train_size:len(apple_close), :]

print("Split into train and test sets" , len(apple_train), len(apple_test))

# use a period of days to predict the prices of stock. Ex: look back the prices for last days and predict for future.

def create_ts(ds, series):
    X, Y =[], []
    for i in range(len(ds)-series - 1):
        item = ds[i:(i+series), 0]
        X.append(item)
        Y.append(ds[i+series, 0])
    return np.array(X), np.array(Y)


series = 7
train_X, train_Y = create_ts(apple_train, series)
test_X, train_Y = create_ts(apple_test, series)

#reshape into  LSTM format - samples, steps, features
train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))

test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))


#Step 2 Build Model
model = Sequential()

model.add(LSTM(
    input_dim=1,
    output_dim=50,
    return_sequences=True))
model.add(Dropout(0.2))

# model.add(LSTM(
#     100,
#     return_sequences=False))
# model.add(Dropout(0.2))

# model.add(Dense(
#     output_dim=1))
# model.add(Activation('linear'))
#
# start = time.time()
# model.compile(loss='mse', optimizer='rmsprop')
# print('compilation time : ', time.time() - start)










