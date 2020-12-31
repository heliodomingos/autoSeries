import itertools
import statsmodels.api as sm
import numpy as np
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import streamlit as st
import time

def false_arima(time_series):
    st.write('..........................ARIMA Falso..........................')
    percentagem = 0.95
    t = time.time()
    train_data, test_data = time_series[0:int(len(time_series) * percentagem)], time_series[int(len(time_series) * percentagem):]
    training_data = train_data[time_series.columns[0]].values
    test_data = test_data[time_series.columns[0]].values
    history = [x for x in training_data]
    model_predictions = []
    N_test_observations = len(test_data)
    for time_point in range(N_test_observations):
        model = ARIMA(history, order=(4, 1, 0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        model_predictions.append(yhat)
        true_test_value = test_data[time_point]
        history.append(true_test_value)
    MSE_error = mean_squared_error(test_data, model_predictions)
    print('Testing Mean Squared Error is {}'.format(MSE_error))

    st.write('time:', time.time()-t)
    st.write('Test Mean Absolute Error:', mean_absolute_error(test_data, model_predictions))
    st.write('Test Root Mean Squared Error:', np.sqrt(mean_squared_error(test_data, model_predictions)))


    test_data = np.array(test_data)
    model_predictions = np.array(model_predictions).transpose()[0]
    A = pd.DataFrame({'Actual Data': test_data, 'Predictions': model_predictions, 'Diff': test_data - model_predictions},
                     index=np.array(time_series.index[int(len(time_series) * percentagem):]))
    st.line_chart(A)
    return 0


def arimaa(time_series):
    return false_arima(time_series)
    st.write('..........................ARIMA..........................')
    t = time.time()
    percentagem = 0.95
    train_data, test_data = time_series[0:int(len(time_series) * percentagem)], time_series[int(len(time_series) * percentagem):]
    training_data = train_data[time_series.columns[0]].values
    test_data = test_data[time_series.columns[0]].values
    history = [x for x in training_data]
    model_predictions = []
    N_test_observations = len(test_data)
    for time_point in range(N_test_observations):
        model = ARIMA(history, order=(4, 1, 0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        model_predictions.append(yhat)
        true_test_value = test_data[time_point]
        history.append(yhat)
    MSE_error = mean_squared_error(test_data, model_predictions)
    print('Testing Mean Squared Error is {}'.format(MSE_error))

    st.write('time:', time.time()-t)
    st.write('Test Mean Absolute Error:', mean_absolute_error(test_data, model_predictions))
    st.write('Test Root Mean Squared Error:', np.sqrt(mean_squared_error(test_data, model_predictions)))


    test_data = np.array(test_data)
    model_predictions = np.array(model_predictions).transpose()[0]
    A = pd.DataFrame({'Actual Data': test_data, 'Predictions': model_predictions, 'Diff': test_data - model_predictions},
                     index=np.array(time_series.index[int(len(time_series) * percentagem):]))
    st.line_chart(A)
    return false_arima(time_series)


def arima_model(series, path_to_save_fig):
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    print('Examples of parameter combinations for Seasonal ARIMA...')
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
    lowest = 0
    bestParam = pdq[0]
    bestParamSeasonal = seasonal_pdq[0][0]
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(series,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
                results = mod.fit()
                if results.aic < lowest:
                    lowest = results.aic
                    bestParam = param
                    bestParamSeasonal = param_seasonal
                #print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
            except:
                continue
    mod = sm.tsa.statespace.SARIMAX(series,
                                    order=bestParam,
                                    seasonal_order=bestParamSeasonal,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    results = mod.fit()
    print(results.summary().tables[1])
    return 0
