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
import sys


app = Flask(__name__)
Bootstrap(app)

@app.route('/')
def index():
    return render_template('index.html')



def fetch_data(companyname, start, end):
    return pdb.DataReader(companyname, 'yahoo', start, end)


def get_stock_data(companyname):

    print("Fetching stock prices to train model ", companyname)

    #Starting date provided. We are taking 1 year data as. of now
    start = datetime.datetime(2018, 1, 1)
    end = datetime.datetime(2018, 11, 1)
    data = fetch_data(companyname, start, end)

    return data


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

    #print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

    # Prophet plots the observed values of our time series (the black dots), the forecasted values (blue line) and
    # the uncertainty intervalsof our forecasts (the blue shaded regions).

    #forecast_plot = clf.plot(forecast)
    #forecast_plot.show()

    # make the vizualization a little better to understand
    stock_data.set_index('ds', inplace=True)
    forecast.set_index('ds', inplace=True)
    # date = df['ds'].tail(plot_num)

    stock_visual = stock_data.join(forecast[['yhat', 'yhat_lower', 'yhat_upper']], how='outer')
    # Visualize the original values .. non logarithmic.
    stock_visual['yhat_scaled'] = np.exp(stock_visual['yhat'])

    actual_data = stock_visual.Close.apply(lambda x: round(x, 2))

    forecasted_data = stock_visual.yhat_scaled.apply(lambda x: round(x, 2))

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
    df = df[['Date', 'Forecasted_Price']]
    df = df[-num_days:]
    df['Date_new'] = pd.to_datetime(df['Date'])
    df['Future_date'] = df['Date_new'].dt.strftime('%Y-%m-%d')
    df = df[['Future_date', 'Forecasted_Price']]
    df.set_index('Future_date', inplace=True)
    suggestion = "Buy" if df['Forecasted_Price'][0] > ending_stock_price else "Sell"

    return graph_data, df, suggestion, d

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
    #print(svm)

    graph = pygal.Line()
    graph.title = '%SVM Model%'
    graph.x_labels = svm.index
    graph.add('Actual data', svm['Close'])
    graph.add('Forecasted data', svm['predictions'])
    graph_data = graph.render_data_uri()
    suggestion = "Buy" if svm_future['predictions'][0] > svm_original['Close'][len(svm_original) - 1] else "Sell"

    #Refactoring done for UI
    svm_future.index.names = ['Future_date']
    svm_future = svm_future.rename(columns={'Close': 'Actual_Price', 'predictions': 'Forecasted_Price'})
    del svm_future['Actual_Price']

    return graph_data, svm_future, suggestion, svm


def prepare_data(df,forecast_col,forecast_out,test_size):

    label = df[forecast_col].shift(-forecast_out);#creating new column called label with the last 5 rows are nan
    X = np.array(df[[forecast_col]]); #creating the feature array

    X = preprocessing.scale(X) #processing the feature array
    X_lately = X[-forecast_out:] #creating the column i want to use later in the predicting method
    X = X[:-forecast_out] # X that will contain the training and testing


    label.dropna(inplace=True); #dropping na values
    y = np.array(label)  # assigning Y
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size) #cross validation

    response = [X_train,X_test , Y_train, Y_test , X_lately, X]
    return response

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

    # calculate mean square error
    error_df = final_df.dropna()

    graph = pygal.Line()
    graph.title = '%Linear Model%'
    graph.x_labels = final_df['Date']
    graph.add('Actual data', final_df['Close'])
    graph.add('Forecasted data', final_df['Act_forecast'])
    graph_data = graph.render_data_uri()

    #Refactoring for UI - part of integration
    linear = linear.rename(columns={'Date': 'Future_date', 'Act_forecast': 'Forecasted_Price'})
    del linear['Close']
    linear.set_index('Future_date', inplace=True)

    return graph_data,linear , suggestion, error_df


@app.route('/predict',methods=['POST','GET'])
def predict():

 if request.method == 'POST':

    #Get stock history from yahoo page
    try:
        companyname = request.form['companyname']
        num_days = int(request.form['num_days'])
        print("Getting data for" + companyname)

        stockhistory = get_stock_data(companyname)

        graph_data, df, suggestion_pro, d = prophet_model(stockhistory, num_days)
        graph_data_svm, df2, suggestion_svm, svm = svm_model(stockhistory, num_days)
        graph_data_linear, linear, suggestion_linear, error_df = linear_regression(stockhistory, num_days)

        return render_template("graphing.html", graph_data=graph_data, graph_data_svm=graph_data_svm,
                               graph_data_linear=graph_data_linear, \
                               tables=[df.to_html()], svm_table=[df2.to_html()], linear_table=[linear.to_html()], \
                               suggestion_pro=suggestion_pro, suggestion_svm=suggestion_svm,
                               suggestion_linear=suggestion_linear)

    except Exception as e:
        return render_template("404.html", error=e)


if __name__ == "__main__":
    app.run(debug=True, threaded=True)