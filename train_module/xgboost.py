import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import streamlit as st
import time

os.environ["PATH"] += os.pathsep + "C:/Users/helio/Anaconda3/Lib/site-packages/graphviz"
# "C:/Users/helio/Anaconda3/pkgs/python-graphviz-0.14-py_0/site-packages/graphviz"

# 'D:/Program Files (x86)/Graphviz2.38/bin/'
#plt.style.use('fivethirtyeight')


def create_train_validation_test(series):
    size = len(series)
    train_size = int(size * 0.8)
    train = series.iloc[:train_size].copy()
    train.columns = series.columns
    return train, series.iloc[train_size:].copy()



def xg_boost(time_series, series_name, index, params):


    '''min_i_mae = 0
    min_mae = 50000000
    min_i_rmse = 500000
    min_rmse = 500000
    for i in [0, 0.25, 0.50, 0.75]:
        s = partition_xg_boost(time_series.copy(), series_name, index, i)
        import streamlit as st
        st.write(index, s[1], s[2], "linha inicial: ", time_series.shape[0], "linha final:",
                 time_series[int(time_series.shape[0] * i):].shape[0])
        if s[1] < min_mae:
            min_i_mae = i
            min_mae = s[1]
        if s[2] < min_rmse:
            min_i_rmse = i
            min_rmse = s[2]
    st.write("best: ", min_i_mae, " ", min_mae, " ", min_i_rmse, " ", min_rmse)
    '''


    original = time_series
    from sklearn.preprocessing import MinMaxScaler
    time_series = time_series.copy()

    # time_series['date'] = time_series.index.astype(int)

    dataset = time_series[list(set(time_series.columns) - {time_series.columns[0]})].values
    features_name = list(set(time_series.columns) - {time_series.columns[0]})
    dataset = dataset.astype('float32')

    dates = time_series[time_series.columns[0]]
    train_size = int(len(dataset) * 0.9)
    test_size = len(dataset) - train_size
    X_train, X_test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    y_train, y_test = dates[0:train_size], dates[train_size:len(dataset)]

    if params == {}:
        reg = xgb.XGBRegressor(n_estimators=1000)
    else:
        reg = xgb.XGBRegressor(n_estimators=params['n_estimators'],
                                                max_depth=params['max_depth'])  # ,
    #                      features_name=features_name)
    reg.fit(X_train, y_train,
            eval_set=[(X_train, y_train)],  # , (X_test, y_test)],
            early_stopping_rounds=50,
            verbose=False)  # Change verbose to True if you want to see it train

    dates = np.array(time_series.index)
    size = len(time_series)
    train_size = int(size * 0.9)
    test_dates = dates[train_size:size]
    test_dates = np.reshape(test_dates, (test_dates.shape[0], 1, 1))
    pred = reg.predict(X_test)

    reg.get_booster().feature_names = features_name

    def cal_accuracy(y_test, pred, last_train_value):
        r = []
        r_v = []
        eee_last = last_train_value
        for e in y_test:
            if e - eee_last > 0:
                r.append(1)
            else:
                r.append(0)
        for j in pred:
            if j - eee_last > 0:
                r_v.append(1)
            else:
                r_v.append(0)
        from sklearn.metrics import accuracy_score
        return accuracy_score(r, r_v)

    def my_plot_importance(model, importance_type):
        ax = plot_importance(model, importance_type=importance_type, max_num_features=10)
        # ax.set_title("Feature Importance - ", importance_type)
        ax.figure.savefig('report/' + series_name + "/" + index + '_' + importance_type + '_feature_importance.png',
                          bbox_inches='tight')

    my_plot_importance(reg, "cover")
    my_plot_importance(reg, "weight")
    my_plot_importance(reg, "gain")

    import shap

    # load JS visualization code to notebook
    #shap.initjs()
    #shap_values = shap.TreeExplainer(reg).shap_values(X_train)
    #shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
    #plt.savefig('report/' + series_name + '/' + index + '_shap_value.png')

    # explain the model's predictions using SHAP
    # (same syntax works for LightGBM, CatBoost, scikit-learn and spark models)
    #explainer = shap.TreeExplainer(reg)
    #shap_values = explainer.shap_values(X_train)
    # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
    # shap.force_plot(explainer.expected_value, shap_values[0, :], time_series.iloc[0:train_size, 1:], matplotlib=True, show=False)
    # visualize the training set predictions
    #shap.force_plot(explainer.expected_value, shap_values, X_train, show=False)
    #plt.savefig(series_name + '_d.png')
    ##########################
    #       SHAP FEATURE IMPORTANCE
    ##########################
    #plt.cla()
    #explainer = shap.TreeExplainer(reg)
    #shap_values = explainer.shap_values(X_train)
    #np.abs(shap_values.sum(1) + explainer.expected_value - pred).max()
    #shap.summary_plot(shap_values, X_train, show=False)
    #plt.savefig(series_name + "_e.png")
    ##############################
    # Fit model using each importance as a threshold
    features_names = time_series.columns[1:]
    features_threshold = reg.feature_importances_
    features_names, features_threshold = (list(t) for t in zip(*sorted(zip(features_names, features_threshold))))

    A = pd.DataFrame({'Actual Data': y_test, 'Predictions': pred},  # 'Diff': y_test - pred},
                     index=np.array(time_series.index[train_size:len(dataset)]))
    ###########################################
    #          Correlation                    #
    #              Plot                       #
    ###########################################
    corr = np.corrcoef(y_test, pred)[0][1]
    st.write("corrrr", index, corr)
    B = pd.DataFrame({'Actual Data': y_test, 'Predictions': pred})
    fig = B.plot.scatter(x='Actual Data',
                         y='Predictions',
                         c='DarkBlue').get_figure()
    fig.savefig('report/' + series_name + "/" + index + '_correlation.png')

    ###########################################
    #          Correlation                    #
    #              Plot                       #
    ###########################################
    import scipy.stats
    slope, intercept, r, p, stderr = scipy.stats.linregress(y_test, pred)
    line = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r={r:.2f}'
    st.write(line)
    fig, ax = plt.subplots()
    ax.plot(y_test, pred, linewidth=0, marker='o', label='Data points', c='DarkBlue')
    ax.plot(y_test, intercept + slope * y_test, label=line, c='orange')
    ax.set_xlabel('Actual Data')
    ax.set_ylabel('Predictions')
    ax.legend(facecolor='white')
    plt.savefig('report/' + series_name + "/" + index + '_regression_line.png')
    st.image('report/' + series_name + "/" + index + '_regression_line.png')
    return {
        'Image1': 'report/' + series_name + "/" + index + '_cover_feature_importance.png',
        'Image2': 'report/' + series_name + "/" + index + '_weight_feature_importance.png',
        'Image3': 'report/' + series_name + "/" + index + '_gain_feature_importance.png',
        'Test Mean Absolute Error': mean_absolute_error(y_test, pred),
        'Test Root Mean Squared Error': np.sqrt(mean_squared_error(y_test, pred)),
        'Test Mean Absolute Percentage Error': np.mean(np.abs((y_test - pred) / y_test)) * 100,
        'Predictions': A,
        'graphviz': 'report/' + series_name + "/" + index + '_graphviz.png',
        'features_name': features_names,
        'model': reg,
        'accuracy': cal_accuracy(y_test, pred, y_train[-1]),
        'correlation': corr
    }


def xg_boost_validation(time_series, series_name, index, split=0):
    ###################################
    ###################################
    n_estimators = 100  # Number of boosted trees to fit. default = 100
    max_depth = 3  # Maximum tree depth for base learners. default = 3
    learning_rate = 0.1  # Boosting learning rate (xgb’s “eta”). default = 0.1
    min_child_weight = 1  # Minimum sum of instance weight(hessian) needed in a child. default = 1
    subsample = 1  # Subsample ratio of the training instance. default = 1
    colsample_bytree = 1  # Subsample ratio of columns when constructing each tree. default = 1
    colsample_bylevel = 1  # Subsample ratio of columns for each split, in each level. default = 1
    gamma = 0  # Minimum loss reduction required to make a further partition on a leaf node of the tree. default=0

    colors = [
        '#1f77b4',  # muted blue
        '#ff7f0e',  # safety orange
        '#2ca02c',  # cooked asparagus green
        '#d62728',  # brick red
        '#9467bd',  # muted purple
        '#8c564b',  # chestnut brown
        '#e377c2',  # raspberry yogurt pink
        '#7f7f7f',  # middle gray
        '#bcbd22',  # curry yellow-green
        '#17becf'  # blue-teal
    ]

    ###################################
    ###################################
    from collections import defaultdict

    from matplotlib import pyplot as plt

    from sklearn.metrics import mean_squared_error

    from tqdm import tqdm_notebook
    from train_module.xg_aux import get_error_metrics

    param_label = 'n_estimators'
    param_list = range(1, 1000, 10)

    param2_label = 'max_depth'
    param2_list = [2, 3, 4, 5, 6, 7, 8, 9]

    error_rate = defaultdict(list)

    tic = time.time()

    for param in tqdm_notebook(param_list):
        for param2 in param2_list:
            # rmse_mean, mape_mean, mae_mean, _
            error_dict = hyperparameter_tunning(time_series,
                                                N=1,
                                                n_estimators=param,
                                                max_depth=param2,
                                                learning_rate=learning_rate,
                                                min_child_weight=min_child_weight,
                                                subsample=subsample,
                                                colsample_bytree=colsample_bytree,
                                                colsample_bylevel=colsample_bylevel,
                                                gamma=gamma)

            # Collect results
            error_rate[param_label].append(param)
            error_rate[param2_label].append(param2)
            error_rate['rmse'].append(error_dict['RMSE'])
            error_rate['mape'].append(error_dict['MAPE'])
            error_rate['mae'].append(error_dict['MAE'])

    error_rate = pd.DataFrame(error_rate)
    toc = time.time()
    print("Minutes taken = {0:.2f}".format((toc - tic) / 60.0))
    print("error_Rate: ", error_rate)

    # Get optimum value for param and param2, using RMSE
    temp = error_rate[error_rate['rmse'] == error_rate['rmse'].min()]
    n_estimators_opt = temp['n_estimators'].values[0]
    max_depth_opt = temp['max_depth'].values[0]
    print("min RMSE = %0.3f" % error_rate['rmse'].min())
    print("optimum params = ", n_estimators_opt, max_depth_opt)
    return {'n_estimators': n_estimators_opt, 'max_depth': max_depth_opt}


def hyperparameter_tunning(time_series, N, n_estimators=100, max_depth=3, learning_rate=0.1,
                           min_child_weight=1, subsample=1, colsample_bytree=1, colsample_bylevel=1, gamma=0):
    time_series = time_series[:len(time_series) * N]
    # time_series = time_series.copy()

    dataset = time_series[list(set(time_series.columns) - {time_series.columns[0]})].values
    features_name = list(set(time_series.columns) - {time_series.columns[0]})
    dataset = dataset.astype('float32')

    dates = time_series[time_series.columns[0]]
    train_size = int(len(dataset) * 0.9)
    X_train, X_test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    y_train, y_test = dates[0:train_size], dates[train_size:len(dataset)]

    reg = xgb.XGBRegressor(objective='reg:squarederror',
                           seed=100,
                           n_estimators=n_estimators,
                           max_depth=max_depth,
                           learning_rate=learning_rate,
                           min_child_weight=min_child_weight,
                           subsample=subsample,
                           colsample_bytree=colsample_bytree,
                           colsample_bylevel=colsample_bylevel,
                           gamma=gamma)
    reg.fit(X_train, y_train,
            eval_set=[(X_train, y_train)],
            early_stopping_rounds=50,
            verbose=False
            )  # Change verbose to True if you want to see it train

    pred = reg.predict(X_test)

    return {
        'MAE': mean_absolute_error(y_test, pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, pred)),
        'MAPE': np.mean(np.abs((y_test - pred) / y_test)) * 100,
        'accuracy': cal_accuracy(y_test, pred, y_train[-1]),
        'correlation': np.corrcoef(y_test, pred)
    }


def cal_accuracy(y_test, pred, last_train_value):
    r = []
    r_v = []
    eee_last = last_train_value
    for e in y_test:
        if e - eee_last > 0:
            r.append(1)
        else:
            r.append(0)
    for j in pred:
        if j - eee_last > 0:
            r_v.append(1)
        else:
            r_v.append(0)
    from sklearn.metrics import accuracy_score
    return accuracy_score(r, r_v)
