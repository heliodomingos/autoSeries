from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

import train_module
from feature_engineering_module.moving_average import *
from train_module.arima import arima_model
from train_module.sarima import sarima_model
from train_module.LSTM import myLSTM
from train_module.LinearReg import lreg
from train_module.xgboost import xg_boost
from forecasting_module.prophet import *
from sklearn.linear_model import LinearRegression
from forecasting_module.linear_regression import *
from feature_engineering_module.tsfresh_gen import *
from feature_engineering_module.pattern_mining import *
from statsmodels.tsa.ar_model import AutoReg
from error_functions_definition import mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from statsmodels.tsa.arima_model import ARIMA
from matrixprofile import *
from sklearn.metrics import classification_report

import streamlit as st
import seaborn as sns
import pandas as pd

sns.set_context("paper", font_scale=1.3)
sns.set_style('white')


def baseline_LSTM(original, time_series, series_name, n_diffs, z):
    return myLSTM(original, time_series, series_name, n_diffs, z)


def create_xg_boost(time_series, series_name):
    return xg_boost(time_series, series_name)


def compute_arima(series, path_to_save_fig):
    return arima_model(series, path_to_save_fig)


def compute_sarima(train, validation):
    return sarima_model(train, validation)


def linear_regression(time_series):
    return lreg(time_series)


def create_train_validation_test(series):
    size = len(series)
    train_size = int(size * 0.7)
    validation_size = int(size * 0.2)
    train = series.iloc[:train_size]
    train.columns = series.columns
    return train, series.iloc[train_size:train_size + validation_size], series.iloc[
                                                                        train_size + validation_size:]


def train_series_(series):
    from statsmodels.tsa.ar_model import AutoReg
    from random import random
    # contrived dataset
    train, validation, test = create_train_validation_test(series)

    index_to_predict = np.array(validation.index).reshape(-1, 1)
    l = train.columns
    train = train.interpolate(method='polynomial', order=2)
    train = train.dropna()

    rd_pred = use_random_forest(train, index_to_predict)
    print("Random forest map: ", mean_absolute_percentage_error(rd_pred, validation.to_numpy()))
    st.write('Using random forest to predict')
    st.write('Mean absolute Percentage Error: ', mean_absolute_percentage_error(rd_pred, validation.to_numpy()))
    # st.write(use_matrix_profile(train_series[key], index_to_predict, key))
    # model = ARIMA(train_series[key]['CLOSE'], order=(0, 1, 0))
    # model_fit = model.fit()
    # outcome = model_fit.forecast()[:len(validation_series[key])]
    # print("ARIMA")
    # print("map: ", mean_absolute_percentage_error(outcome.to_numpy(),validation_series[key]['CLOSE'].to_numpy()))
    # print("--ARIMA")
    # model = AutoReg(train_series[key]['CLOSE'], lags=2)
    # model_fit = model.fit()
    # prediction = model_fit.predict(start=len(train_series[key]), end=(len(train_series[key])+len(validation_series[key]))-1)
    # print("map: ", mean_absolute_percentage_error(prediction.to_numpy(),validation_series[key]['CLOSE'].to_numpy()))

    # print(train_series[key])
    # cv_score = cross_val_score(LogisticRegression(),
    #                           train_series[key].index.values, train_series[key]['CLOSE'].values,
    #                           scoring='accuracy',
    #                           cv=3,
    #                           n_jobs=-1,
    #                           verbose=1)
    # print(cv_score)

    # print(model.predict(validation_series[key].index.values))


def train_series_old(dict_of_series, target_variable):
    train_series = {}
    validation_series = {}
    test_series = {}
    for key in dict_of_series:
        if dict_of_series[key].isnull().values.any():
            dict_of_series[key] = dict_of_series[key].interpolate()
        train_series[key], validation_series[key], test_series[key] = create_train_validation_test(dict_of_series[key])

    for key in train_series:
        index_to_predict = np.array(validation_series[key].index).reshape(-1, 1)
        l = train_series[key].columns
        train_series[key] = train_series[key].interpolate(method='polynomial', order=2)
        train_series[key] = train_series[key].dropna()
        print(key)
        rd_pred = use_random_forest(train_series[key], index_to_predict)
        print("Random forest map: ", mean_absolute_percentage_error(rd_pred, validation_series[key].to_numpy()))

        use_matrix_profile(train_series[key], index_to_predict)
        # model = ARIMA(train_series[key]['CLOSE'], order=(0, 1, 0))
        # model_fit = model.fit()
        # outcome = model_fit.forecast()[:len(validation_series[key])]
        # print("ARIMA")
        # print("map: ", mean_absolute_percentage_error(outcome.to_numpy(),validation_series[key]['CLOSE'].to_numpy()))
        # print("--ARIMA")
        # model = AutoReg(train_series[key]['CLOSE'], lags=2)
        # model_fit = model.fit()
        # prediction = model_fit.predict(start=len(train_series[key]), end=(len(train_series[key])+len(validation_series[key]))-1)
        # print("map: ", mean_absolute_percentage_error(prediction.to_numpy(),validation_series[key]['CLOSE'].to_numpy()))

        # print(train_series[key])
        # cv_score = cross_val_score(LogisticRegression(),
        #                           train_series[key].index.values, train_series[key]['CLOSE'].values,
        #                           scoring='accuracy',
        #                           cv=3,
        #                           n_jobs=-1,
        #                           verbose=1)
        # print(cv_score)

        # print(model.predict(validation_series[key].index.values))
        break


def use_random_forest(time_series, n_predictions):
    print("RandomForest: Begin")
    model = RandomForestRegressor(n_estimators=1000, n_jobs=-1, random_state=0)
    model = model.fit(np.array(time_series.index).reshape(-1, 1), time_series.values)
    predictions = model.predict(n_predictions)
    print("RandomForest: End")
    return predictions


def use_matrix_profile(time_series, n_predictions, target_variable):
    print("Matrix Profile")
    return matrixProfile.stomp(time_series[target_variable].values, 4)
