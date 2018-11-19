import pandas as pd
import numpy as np
import pandas_datareader.data as pdb
from fbprophet import Prophet
import datetime
from flask import Flask, render_template, url_for , make_response
from flask import request, redirect
from flask_bootstrap import Bootstrap
import csv
from itertools import zip_longest
import pygal
from sklearn.svm import SVR
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def prophet_model(stockhistory,num_days):
    stock_data = stockhistory.filter(['Close'])

    # Prophet would need a feature column ds. Hence creating a new column with name ds which is the Date feature.
    stock_data['ds'] = stock_data.index

    # log transform the ‘Close’ variable to convert non-stationary data to stationary.
    stock_data['y'] = np.log(stock_data['Close'])

    # Using the Prophet model for analysis
    clf = Prophet()
    clf.fit(stock_data)

    ending_stock_price = stock_data['Close'][-1]

    # num_days = 10
    future = clf.make_future_dataframe(periods=num_days)
    forecast = clf.predict(future)

    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

    # Prophet plots the observed values of our time series (the black dots), the forecasted values (blue line) and
    # the uncertainty intervalsof our forecasts (the blue shaded regions).

    forecast_plot = clf.plot(forecast)
    forecast_plot.show()

    # make the vizualization a little better to understand
    stock_data.set_index('ds', inplace=True)
    forecast.set_index('ds', inplace=True)
    # date = df['ds'].tail(plot_num)

    stock_visual = stock_data.join(forecast[['yhat', 'yhat_lower', 'yhat_upper']], how='outer')
    # Visualize the original values .. non logarithmic.
    stock_visual['yhat_scaled'] = np.exp(stock_visual['yhat'])

    actual_data = stock_visual.Close.apply(lambda x: round(x, 2))

    forecasted_data = stock_visual.yhat_scaled.apply(lambda x: round(x, 2))
    # forecasted_data = "%.2f" % forecasted_data

    ###### Calculate meam square error
    fore = [forecasted_data[i] for i in range(len(actual_data))]
    # print(len(fore))
    # print(len(actual_data))
    se = np.square(fore - actual_data)
    mse = np.mean(se)
    rmse = np.sqrt(mse)
    #print(rmse)


    date = future['ds']

    d = [date, actual_data, forecasted_data]

    # predictions = pd.DataFrame(np.column_stack([date, actual_data, forecasted_data]), columns=['Date', 'Actual_Price','Predicted_Price'])

    readcsvdata = zip_longest(*d, fillvalue='')
    with open('predictions.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(("Date", "Actual_price", "Forecasted_Price"))
        wr.writerows(readcsvdata)
    myfile.close()

    graph = pygal.Line()
    graph.title = '%Prophet Model%'
    graph.x_labels = date
    graph.add('Actual data', actual_data)
    graph.add('Forecasted data', forecasted_data)
    graph_data = graph.render_data_uri()



    df = pd.read_csv('Predictions.csv')

    # calculate mean square error
    # error_df = df.dropna()
    # print(df)
    # mse = np.mean(np.abs((error_df['Actual_Price'] - error_df['Forecasted_Price']) / error_df['Actual_price'])) * 100
    # print("MSE:", mse)
    # rmse = np.sqrt(mse)
    # print("RMSE:", rmse)


    df = df[['Date', 'Forecasted_Price']]
    df = df[-num_days:]
    df['Date_new'] = pd.to_datetime(df['Date'])
    df['Future_date'] = df['Date_new'].dt.strftime('%Y-%m-%d')
    df = df[['Future_date', 'Forecasted_Price']]
    df.set_index('Future_date', inplace=True)
    #suggestion = "Buy" if df['Forecasted_Price'][0] > ending_stock_price else "Sell"

    next_price = df['Forecasted_Price'][0]
    suggestion = "Buy" if ((next_price > ending_stock_price) and (next_price - ending_stock_price) > 1) else "Sell" \
        if ((ending_stock_price > next_price) and (ending_stock_price - next_price) > 1) else "Hold"

    return graph_data, df, suggestion


start = datetime.datetime(2018, 1, 1)
end = datetime.datetime(2018, 11, 12)
stockhistory = pdb.DataReader('AAPL', 'yahoo', start, end)

graph, df1, prophet = prophet_model(stockhistory,1)