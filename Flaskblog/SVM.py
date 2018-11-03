import pygal
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
#import Talib
import random
import datetime
import pandas_datareader.data as pdb
from sklearn import linear_model





def feature(df,date_column):
    df[date_column] = df[date_column].dt.strftime('%Y-%m-%d')
    df['day'] = df[date_column].apply(lambda x: x.split('-')[2]).astype(int)
    df['month'] = df[date_column].apply(lambda x: x.split('-')[1]).astype(int)
    df['year'] = df[date_column].apply(lambda x: x.split('-')[0]).astype(int)
    df = df[['day', 'month', 'year']]
    X= np.array(df)
    dates = np.reshape(X, (len(X), 3))
    return dates

def get_features(df,date_column,ylabel):

    df[date_column] = df[date_column].dt.strftime('%Y-%m-%d')
    df['day'] = df[date_column].apply(lambda x: x.split('-')[2]).astype(int)
    df['month'] = df[date_column].apply(lambda x: x.split('-')[1]).astype(int)
    df['year'] = df[date_column].apply(lambda x: x.split('-')[0]).astype(int)
    df = df[['day', 'month', 'year', ylabel]]

    ## Preparing Feature matrix
    X = np.array(df.drop([ylabel], 1))
    ## Preparing label
    prices = []
    for row in df[ylabel]:
        prices.append(float(row))

    dates = np.reshape(X, (len(X), 3))  # converting to matrix of n X 1
    #print(dates)
    prices = np.reshape(prices, (len(prices), 1))
    #print(prices)


    return dates, prices

def svm_model(stockhistory,num_days):
    df = stockhistory
    df['ds'] = df.index
    last_date = df.index[-1]


    ### Make future dataframe
    df1 = pd.DataFrame(index=range(num_days), columns=['Date', 'Close', 'predictions'])
    for i in range(num_days):
        df1['Date'][i] = last_date + datetime.timedelta(days=i + 1)

    # print(df1)

    # Get predictions on original dataset
    def create_dataset(df, prediction):
        se = pd.Series(prediction)
        df['predictions'] = se.values

        return df


    dates, prices = get_features(df, 'ds', 'Close')
    # print(dates)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_rbf.fit(dates, prices.ravel())
    prediction = svr_rbf.predict(dates)
    svm_original = create_dataset(df, prediction)
    svm_original = svm_original[['Close', 'predictions']]
    # print(svm_original)

    future_dates = feature(df1, 'Date')
    prediction_new = svr_rbf.predict(future_dates)
    # print(prediction_new)
    svm_future = create_dataset(df1, prediction_new)
    svm_future.set_index('Date', inplace=True)
    svm_future = svm_future[['Close', 'predictions']]
    # print(svm_future)

    svm = svm_original.append(svm_future)
    print(svm)

    graph = pygal.Line()
    graph.title = '%SVM Model%'
    graph.x_labels = svm.index
    graph.add('Actual data', svm['Close'])
    graph.add('Forecasted data', svm['predictions'])
    graph_data = graph.render_data_uri()
    suggestion = "Buy" if svm_future['predictions'][0] > svm_original['Close'][len(svm_original) - 1] else "Sell"
    print("Predicted " + str(svm_future['predictions'][0]))
    print("Actual " + str(svm_original['Close'][len(svm_original) - 1]))
    print(suggestion)
    svm_future.index.names = ['Future_date']

    svm_future = svm_future.rename(columns={'Close':'Actual_Price', 'predictions':'Forecasted_Price'})

    return graph_data, svm_future, suggestion

# # plt.scatter(dates, prices, color='black', label='Actual Data')
# plt.plot(svm.index, svm['Close'], color='red', label='Actual')
# plt.plot(svm.index, svm['predictions'], color='blue', label='Forecasted')
# plt.xlabel('Date')
# plt.ylabel('Price')
# #plt.title('SVM')
# plt.legend()
# plt.show()
#
start = datetime.datetime(2018, 1, 1)
end = datetime.datetime(2018, 11, 2)

df = pdb.DataReader('GOOGL', 'yahoo', start, end)
graph, svm, suggestion = svm_model(df,5)
print(svm)