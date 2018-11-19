import pandas_datareader.data as pdb
import datetime
from sklearn.metrics import mean_squared_error
import math
import pandas as pd
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import style
import pygal

style.use('ggplot')
from sklearn import cross_decomposition
from sklearn.linear_model import LinearRegression
# start=datetime.datetime(2018,1,1)
# end=datetime.datetime.today()
# #print('Today', end)
# stock='AAPL'
#
#
# df=web.DataReader(stock,'yahoo',start,end)
#print(df.tail(5))

def prepare_data(df,forecast_col,forecast_out,test_size):
    label = df[forecast_col].shift(-forecast_out)#creating new column called label with the last 5 rows are nan

    X = np.array(df[[forecast_col]]); #creating the feature array
    #print(X[-5:])
    X = preprocessing.scale(X) #processing the feature array
    X_lately = X[-forecast_out:] #creating the column i want to use later in the predicting method
    #print(X_lately)
    X = X[:-forecast_out] # X that will contain the training and testing


    label.dropna(inplace=True); #dropping na values
    y = np.array(label)  # assigning Y
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size) #cross validation

    response = [X_train,X_test , Y_train, Y_test , X_lately, X]
    return response;

def linear_regression(stockhistory, num_days):
    df = stockhistory
    forecast_col = 'Close'
    forecast_out = num_days
    test_size = 0.2

    X_train, X_test, Y_train, Y_test, X_lately, X = prepare_data(df, forecast_col, forecast_out, test_size)
    learner = LinearRegression()

    learner.fit(X_train, Y_train)
    score = learner.score(X_test, Y_test)
    forecast = learner.predict(X)
    forecast_future = learner.predict(X_lately)
    ##### df - for original values prediction
    df['Forecast'] = 0.0
    for i in range(len(forecast)):
        df['Forecast'][i] = forecast[i]

    df['Act_forecast'] = df['Forecast'].shift(forecast_out)
    df['Date'] = df.index
    #### df2 - for future value prediction
    df2 = df[-forecast_out:]
    #print(df2)
    last_date = df2.index[-1]
    df2 = df[-forecast_out:]
    last_date = df2.index[-1]
    df2['future_date'] = last_date

    df2['future_preds'] = 0.0
    df2['Actual'] = np.nan
    preds = []
    for i in range(len(df2)):
        df2['future_date'][i] = last_date + datetime.timedelta(days=i + 1)
    for i in range(len(forecast_future)):
        df2['future_preds'][i] = forecast_future[i]

    # print(forecast_future)
    linear = df2[['future_date', 'Actual', 'future_preds']].reset_index(drop=True)
    linear = linear.rename(columns={'future_date': 'Date', 'Actual': 'Close', 'future_preds': 'Act_forecast'})
    #print(linear)
    original = df[['Date', 'Close', 'Act_forecast']].reset_index(drop=True)
    # print(original)

    final_df = original.append(linear, sort=True)

    # Suggest whether to buy stock or sell. if predicted value is greater for tomorrow then buy
    suggestion = "Buy" if linear['Act_forecast'][0] > stockhistory['Close'][len(stockhistory)-1] else "Sell"
    # next_price = linear['Act_forecast'][0]
    # ending_stock_price = stockhistory['Close'][len(stockhistory) - 1]
    # suggestion = "Buy" if ((next_price > ending_stock_price) and (next_price - ending_stock_price) > 1) else "Sell" \
    #     if ((ending_stock_price > next_price) and (ending_stock_price - next_price) > 1) else "Hold"

    # calculate mean square error
    error_df = final_df.dropna()

    # graph = pygal.Line()
    # graph.title = '%Linear Model%'
    # graph.x_labels = final_df['Date']
    # graph.add('Actual data', final_df['Close'])
    # graph.add('Forecasted data', final_df['Act_forecast'])
    # graph_data = graph.render_data_uri()

    df['Close'].plot(figsize=(15,6), color="red")
    df['Act_forecast'].plot(figsize=(15,6), color="blue")
    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()


    #Refactoring for UI - part of integration
    linear = linear.rename(columns={'Date': 'Future_date', 'Act_forecast': 'Forecasted_Price'})
    del linear['Close']
    linear.set_index('Future_date', inplace=True)

    return linear , suggestion, error_df

start = datetime.datetime(2018, 1, 1)
end = datetime.datetime(2018, 11, 12)
stockhistory = pdb.DataReader('AAPL', 'yahoo', start, end)

df1, linear, error_df = linear_regression(stockhistory,1)
#forecast_dates, first_day = get_forecast_dates()
# print(df1)
# print(stockhistory)
#print(linear)



# future_dataframe['Forecast']=0.0
# for i in range(len(forecast_future)):
#     future_dataframe['Forecast'][i] = forecast_future[i]
#
# print(future_dataframe)

#
# df['Close'].plot(figsize=(15,6), color="red")
# df['Act_forecast'].plot(figsize=(15,6), color="blue")
# plt.legend(loc=4)
# plt.xlabel('Date')
# plt.ylabel('Price')
# plt.show()



