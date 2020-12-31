from load_data_module import load_data, interpret_timestamp
from data_cleaning_module import cleaning_data
from data import forecast_data
import feature_engineering_module as fe
from train_module import *
from reporting_module import *
import warnings
import os
import time
import chart_studio.plotly as py
import math
import matplotlib
import numpy as np
import pandas as pd
from web import WebPage
from draw_data_module import *
from data_profiling_module import get_distribution_hist

warnings.filterwarnings("ignore")
import streamlit as st
from draw_data_module import plot_diff
from pmdarima.arima import ADFTest
from pmdarima.arima import auto_arima


def create_baseline(data, key):
    t = time.time()
    res = baseline_LSTM(
        data[key]['raw_data'].copy(),
        data[key]['insights']['stationaryData'].copy(),
        key,
        data[key]['insights']['diffs'],
        data[key]['insights']['z_multiple'])

    data[key]['baseline'] = {
        'Test Mean Absolute Error': res[0],
        'Test Root Mean Squared Error': res[1],
        'Test Mean Absolute Percentage Error': res[2],
        'Predictions': res[3],
        'time(s)': time.time() - t,
        'accuracy': res[4],
        'correlation': res[5]
    }
    st.write('baseline acc, corr', data[key]['baseline']['accuracy'], data[key]['baseline']['correlation'])
    # fplot_diff(data, key, 'baseline')


def features_with_tsfresh(data, key):
    # Use tsfresh

    st.write('TSFresh feature generation')
    t0 = time.time()
    t = data[key]['raw_data'].columns[0]
    tsfresh_Result = fe.use_tsfresh(data[key]['raw_data'])
    tsfresh_Result.drop('index', axis=1, inplace=True)
    tsfresh_Result = tsfresh_Result.set_index('date')
    print("tsfresh_Result\n")
    print(tsfresh_Result)
    st.write(tsfresh_Result)
    res = xg_boost(tsfresh_Result, key, 'tsfresh')
    data[key]['tsfresh'] = res
    data[key]['tsfresh']['time(s)']: time.time() - t0
    plot_diff(data, key, 'tsfresh')
    webPage.f_add_xg_boost(data, 'tsfresh')


def show_changes_raw_to_stationary(data, key):
    data[key]['raw_data'].rename(columns={'close': 'close_original'}, inplace=True)

    st.line_chart(data[key]['raw_data'].join(time_series))
    data[key]['raw_data'].rename(columns={'close_original': 'close'}, inplace=True)


def call_xg_boost(data, time_series, key, feature_type, params={}):
    time0 = time.time()
    #result = xg_boost(time_series, key, feature_type, params)
    #st.write('before hyper')
    #st.write('Test Mean Absolute Error', result['Test Mean Absolute Error'])
    #st.write('Test Root Mean Squared Error', result['Test Root Mean Squared Error'])
    #st.write('Test Mean Absolute Percentage Error', result['Test Mean Absolute Percentage Error'])
    #st.write('accuracy', result['accuracy'])
    #st.write('correlation', result['correlation'])

#3if feature_type == '00lags':
    #    from train_module.xgboost import xg_boost_validation
    #    params = xg_boost_validation(time_series, key, feature_type, split=0)
    #    print('PARAMS')
    #    print(params)
    result = xg_boost(time_series, key, feature_type, params)
    #st.write('AFTER hyper')
    #st.write('Test Mean Absolute Error', result['Test Mean Absolute Error'])
    #st.write('Test Root Mean Squared Error', result['Test Root Mean Squared Error'])
    #st.write('Test Mean Absolute Percentage Error', result['Test Mean Absolute Percentage Error'])
    #st.write('accuracy', result['accuracy'])
    #st.write('correlation', result['correlation'])
    data[key][feature_type] = result
    data[key][feature_type]['time(s)'] = time.time() - time0
    data[key][feature_type]['dados'] = time_series
    # plot_diff(data, key, feature_type)
    webPage.f_add_xg_boost(data, key, feature_type)
    #reconstruct_time_series(data, key, feature_type)
    return params


def build_autocorrelation(tt_autocorrelation, i, key, r):
    tt_autocorrelation['lag'] = tt_autocorrelation['close'].shift(i)
    tt_autocorrelation.dropna()
    y_test = tt_autocorrelation['close'].values
    pred = tt_autocorrelation['lag'].values

    fig, ax = plt.subplots()
    ax.plot(y_test, pred, linewidth=0, marker='o', label='Data points', c='DarkBlue')
    # ax.plot(y_test, intercept + slope * y_test, label=line, c='orange')
    ax.set_xlabel('y(t)')
    ax.set_ylabel('y(t-' + str(i) + ')')
    ax.legend(facecolor='white')
    plt.savefig('report/' + key + "/_" + str(i) + "_" + r + "_autocorrelation.png")


def get_mov_avg_std(df, col, N):
    """
    Given sa dataframe, get mean and std dev at timestep t using values from t-1, t-2, ..., t-N.
    Inputs
        df         : dataframe. Can be of any length.
        col        : name of the column you want to calculate mean and std dev
        N          : get mean and std dev at timestep t using values from t-1, t-2, ..., t-N
    Outputs
        df_out     : same as df but with additional column containing mean and std dev
    """
    mean_list = df[col].rolling(window=N, min_periods=1).mean()  # len(mean_list) = len(df)
    std_list = df[col].rolling(window=N, min_periods=1).std()  # first value will be NaN, because normalized by N-1

    # Add one timestep to the predictions
    mean_list = np.concatenate((np.array([np.nan]), np.array(mean_list[:-1])))
    std_list = np.concatenate((np.array([np.nan]), np.array(std_list[:-1])))

    # Append mean_list to df
    df_out = df.copy()
    df_out[col + '_mean'] = mean_list
    df_out[col + '_std'] = std_list

    return df_out


def do_scaling(df):
    """
    Do scaling for the adj_close and lag cols
    """
    # df.loc[:, 'close_scaled'] = (df['close'] - df['close_mean']) / df['close_std']
    # for n in range(N, 0, -1):
    #    df.loc[:, 'adj_close_scaled_lag_' + str(n)] = \
    #        (df['adj_close_lag_' + str(n)] - df['adj_close_mean']) / df['adj_close_std']
    #
    #       # Remove adj_close_lag column which we don't need anymore
    #      df.drop(['adj_close_lag_' + str(n)], axis=1, inplace=True)
    df['close'] = (df["close"] - df["close"].mean()) / df["close"].std()
    return df


def draw_line_chart(df, title, loc):
    return 0


if __name__ == '__main__':

    # os.system('streamlit run kddprocess.py')

    webPage = WebPage()
    webPage.create_forecast_section()

    data = forecast_data()

    for key in data.keys():
        webPage.f_load_data(data, key)

        # data[key]['raw_data'] = cleaning_data(data[key]['raw_data'])
        # print("Data cleaned.")

        time_series = data[key]['insights']['stationaryData']
        # time_series = do_scaling(time_series)
        st.write("After scalling")
        st.line_chart(time_series)
        get_distribution_hist(time_series, key, "after_scaling")
        plot_series(time_series, "close", "report/" + key + "/data_after_scaling.png", key + "after transformations")
        show_changes_raw_to_stationary(data, key)

        webPage.f_data_cleaning()
        webPage.f_create_baseline()

        #########################################
        #           WEB PAGE CREATED            #
        #                                       #
        #           STARTING BASELINE           #
        #########################################

        # for i in range(2, 40):
        #   build_autocorrelation(time_series.copy(), i, key, "after")

        try:
            # raise ValueError("Passa à frente")
            create_baseline(data, key)
            webPage.f_add_baseline_entry(data, key)
            plot_series_two_line(data[key]['baseline']['Predictions'], data[key]['baseline']['Predictions'], "close",
                                 "report/" + key + "/baseline_plot.png",
                                 key + " baseline results")
        except:
            print('Baseline not created')
            import math

            data[key]['baseline'] = {
                'Test Mean Absolute Error': math.inf,
                'Test Root Mean Squared Error': math.inf,
                'Predictions': None,
                'time(s)': "0s"
            }

        webPage.f_update_baseline_status()

        #########################################
        #           BASELINE CREATED            #
        #                                       #
        #           AUTO ARIMA                  #
        #########################################

        try:
            raise ValueError('do nothing')
            train_size = int(len(time_series) * 0.9)
            test_size = len(time_series) - train_size
            train = time_series.iloc[:train_size, :].values
            test = time_series.iloc[train_size:, :].values
            model = auto_arima(train, trace=True)
            st.write(model.summary())
            pred = model.predict(n_periods=test_size)

            A = pd.DataFrame({'Actual Data': test.transpose()[0], 'Predictions': pred,
                              'Diff': test.transpose()[0] - pred},
                             index=np.array(time_series.index)[train_size:])
            st.write(A)
            st.line_chart(A)

        except:
            print('Error creating Arima')
            pass

        #########################################
        #                                       #
        #     XGBOOST as SUPERVISED PROBLEM     #
        #                                       #
        #########################################

        # try:
        t0 = time.time()
        time_series_lags = time_series.copy()
        time_series_lags['lag_2'] = time_series['close'].shift(2)
        time_series_lags['lag_3'] = time_series['close'].shift(3)
        time_series_lags['lag_4'] = time_series['close'].shift(4)
        time_series_lags['lag_5'] = time_series['close'].shift(5)
        time_series_lags['lag_6'] = time_series['close'].shift(6)
        time_series_lags['lag_7'] = time_series['close'].shift(7)
        time_series_lags['lag_8'] = time_series['close'].shift(8)
        time_series_lags['lag_9'] = time_series['close'].shift(9)
        time_series_lags['lag_10'] = time_series['close'].shift(10)
        time_series_lags['lag_11'] = time_series['close'].shift(11)
        time_series_lags['lag_20'] = time_series['close'].shift(20)
        time_series_lags['lag_21'] = time_series['close'].shift(21)

        #print(time_series_lags)
        #print("##\n\n\n\n\n\n##")
        #st.write(time_series_lags)
        corr_matrix = time_series_lags.corr()

        st.write(corr_matrix)

        call_xg_boost(data, time_series_lags, key, "lags")

        # except:
        #    print("Error creating xgboost with lag features.")
        #    pass

        #########################################
        #                                       #
        #         XGBOOST AGG TIMESTAMP         #
        #                                       #
        #########################################

        try:
            t0 = time.time()
            time_series_agg = time_series.copy()
            fe.aggregate_timestamp(time_series_agg, interpret_timestamp(time_series.index))
            call_xg_boost(data, time_series_agg, key, "agg_timestamp")
        except:
            print("Error creating xgboost with agg_timestamp features.")
            pass
        #########################################
        #                                       #
        #        XGBOOST ONLY TIMESTAMP         #
        #                                       #
        #########################################

        try:
            # raise ValueError("passa")
            t0 = time.time()
            time_series_only = time_series.copy()
            fe.only_time(time_series_only)
            # from fastai.tabular import add_datepart

            call_xg_boost(data, time_series_only, key, "only_timestamp")
            st.write("Features atuais: ", time_series_only.columns)
        except:
            print("Error creating xgboost with only_timestamp features.")
            pass
        #########################################
        #                                       #
        #        XGBOOST SMOOTH TIMESTAMP       #
        #                                       #
        #########################################

        try:
            # raise ValueError("passa")
            t0 = time.time()
            time_series_smooth = time_series.copy()
            fe.smoothing_timestamp(time_series_smooth, key, alfas=None, target_variable=time_series.columns[0])

            call_xg_boost(data, time_series_smooth, key, 'smooth_timestamp')

        except:
            print("Error creating xgboost with smooth_timestamp features.")
            pass



        #########################################
        #                                       #
        #        XGBOOST Composition            #
        #                                       #
        #########################################

        try:

            # raise ValueError("passa")
            t0 = time.time()
            time_series_composition = data[key]["lags"]["dados"]
            features_to_composition = data[key]["agg_timestamp"]["features_name"]
            fe.aggregate_composition(time_series_composition, features_to_composition)
            call_xg_boost(data, time_series_composition, key, "lagsXXagg_timestamp")

        except:
            print("Error creating xgboost with only_timestamp features.")
            pass

        #########################################
        #                                       #
        #           XGBOOST FINAL               #
        #                                       #
        #########################################

        try:
            # raise ValueError("passa")
            time_series_final = time_series

            for fff in data[key].keys():
                try:
                    for col in data[key][fff]['dados'].columns[1:]:
                        time_series_final[col] = data[key][fff]['dados'][col]

                except:
                    pass
            st.write("Time series final")
            st.write(time_series_final)
            call_xg_boost(data, time_series_final, key, "final")
        except:
            print("Error creating xgboost with only_timestamp features.")
            pass

        #########################################
        #                                       #
        #          XGBOOST WITH DFS             #
        #                                       #
        #########################################
        # features_with_tsfresh(data, key)
        try:
            raise ValueError('just pass')
            t0 = time.time()
            t = data[key]['raw_data'].columns[0]
            fe.dfs(time_series)
            call_xg_boost(data, time_series, key, "dfs")

        except:
            print("Error creating xgboost with dfs features.")
            pass

        try:
            from sklearn.ensemble import BaggingRegressor

            time_seriesv2 = time_series_final.copy()
            st.write("##Bagging")
            # data[key]['final']['dados'] = data[key]['final']['dados'].dropna()
            dataset = time_seriesv2[list(set(time_seriesv2.columns) - {time_seriesv2.columns[0]})].values

            dates = time_seriesv2[time_seriesv2.columns[0]]
            dataset = dataset.astype('float32')
            st.write(dataset)
            train_size = int(len(dataset) * 0.9)
            test_size = len(dataset) - train_size
            X_train, X_test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
            y_train, y_test = dates[0:train_size], dates[train_size:len(dataset)]
            # define dataset
            st.write("Xtrain")
            st.write(X_train)
            st.write("y_train")
            st.write(y_train)
            model = BaggingRegressor()
            # fit the model on the whole dataset
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            A = pd.DataFrame({'Actual Data': y_test, 'Predictions': pred, 'Diff': y_test - pred},
                             index=np.array(data[key]['final']['dados'].index[train_size:len(dataset)]))
            ###########################################
            #          Correlation                    #
            #              Plot                       #
            ###########################################
            corr = np.corrcoef(y_test, pred)[0]
            st.write("Bagging corrrr", index, corr)
            B = pd.DataFrame({'Actual Data': y_test, 'Predictions': pred})
            fig = B.plot.scatter(x='Actual Data',
                                 y='Predictions',
                                 c='DarkBlue').get_figure()
            fig.savefig('report/' + series_name + '/Bagging_correlation.png')
        except:
            pass
        webPage.f_update_feature_status()

        st.write("Results")
        a = {}
        b = {}
        c = {}
        d = {}
        e = {}
        for fff in data[key].keys():
            try:
                a[fff] = data[key][fff]['Test Mean Absolute Error']
                b[fff] = data[key][fff]['Test Root Mean Squared Error']

                d[fff] = data[key][fff]['accuracy']
                e[fff] = data[key][fff]['correlation']
                c[fff] = data[key][fff]['Test Mean Absolute Percentage Error']
            except:
                pass
        st.write(pd.DataFrame([a, b, c, d, e], index=[
            'Test Mean Absolute Error',
            'Test Root Mean Squared Error',
            'Test Mean Absolute Percentage Error',
            'accuracy',
            'correlation']))
        #fig, ((ax1, ax2, ax3), (ax4, ax5)) = plt.subplots(3, 2 )
        #fig.suptitle('Results of the ' + key)
        #ax1 = a.plot.bar(x='Group of Features', y='MAE')
        #ax2 = b.plot.bar(x='Group of Features', y='RMSE')
        #ax3 = c.plot.bar(x='Group of Features', y='MAPE')
        #ax4 = d.plot.bar(x='Group of Features', y='Accuracy')
        #ax5 = e.plot.bar(x='Group of Features', y='Correlation')


        #for ax in fig.get_axes():
        #    ax.label_outer()
        #fig.savefig("report/"+key+"/Results.png")
        #########################################
        #                                       #
        #          VOTING ENSEMBLE              #
        #                                       #
        #########################################
        try:
            raise ValueError("pass")
            st.write("##Voting")
            models = []
            for x in data[key].keys():
                try:

                    models.append(data[key][x]['model'])
                except:
                    pass

            from sklearn.ensemble import VotingRegressor

            st.write("voting models: ", models)
            ensemble = VotingRegressor(estimators=models)
            ensemble.fit(X_train, y_test)
            pred = ensemble.predict(X_test)
            A = pd.DataFrame({'Actual Data': y_test, 'Predictions': pred, 'Diff': y_test - pred},
                             index=np.array(data[key]['final']['dados'].index[train_size:len(dataset)]))
            ###########################################
            #          Correlation                    #
            #              Plot                       #
            ###########################################
            corr = np.corrcoef(y_test, pred)[0]
            st.write("Voting corrrr", index, corr)
            B = pd.DataFrame({'Actual Data': y_test, 'Predictions': pred})
            fig = B.plot.scatter(x='Actual Data',
                                 y='Predictions',
                                 c='DarkBlue').get_figure()
            fig.savefig('report/' + series_name + '/Voting_correlation.png')
        except:
            pass

        st.markdown("<hr>", unsafe_allow_html=True)
