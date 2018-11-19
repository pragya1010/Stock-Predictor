import pandas_datareader.data as pdb
import datetime
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


from app import svm_model, prophet_model, linear_regression


start = datetime.datetime(2018, 1, 1)
end = datetime.datetime(2018, 11, 12)

df = pdb.DataReader('AAPL', 'yahoo', start, end)

def test_svm_model(df):
    graph_data_svm, df2, suggestion_svm, svm = svm_model(df, 1)

    svm = svm.dropna()
    # print(svm)

    se = np.square(svm['Close'] - svm['predictions'])
    mse = np.mean(se)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((svm['Close'] - svm['predictions']) / svm['Close'])) * 100

    #print("Root mean square error for Prophet Model:" + str(rmse)) # For Apple data - 3.174
    #print(mape)
    return rmse, mape

def test_prophet_model(df):
    graph_data, df, suggestion_pro, d = prophet_model(df, 1)

    actual_data = np.array(d[1].reset_index(drop = True).dropna())
    forecasted_data = np.array(d[2].reset_index(drop = True).dropna())
    fore = [forecasted_data[i] for i in range(len(actual_data))]

    se = np.square(fore - actual_data)
    mse = np.mean(se)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual_data - fore) / actual_data)) * 100

    return rmse, mape

    #print("Root mean square error for SVM Model:" + str(rmse)) # For Apple data - 2.23

def test_linear_model(df):
    graph_data, linear, suggestion, error_df = linear_regression(df, 1)
    se = np.square(error_df['Close'] - error_df['Act_forecast'])
    mse = np.mean(se)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((error_df['Close'] - error_df['Act_forecast']) / error_df['Close'])) * 100

    return rmse, mape


rmse_linear, mape_linear = test_linear_model(df)
rmse_prophet, mape_prophet = test_prophet_model(df)
rmse_svm, mape_svm = test_svm_model(df)



print("------Prophet Model------")
print("RMSE:{}  MAPE:{}" .format(str(rmse_prophet),str(mape_prophet)))


print("------SVM Model------")
print("RMSE:{}  MAPE:{}" .format(str(rmse_svm),str(mape_svm)))


print("------Linear Model------")
print("RMSE:{}  MAPE:{}" .format(str(rmse_linear),str(mape_linear)))





