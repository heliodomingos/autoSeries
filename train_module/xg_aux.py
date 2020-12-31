import numpy as np
from xgboost import XGBRegressor
import math


def get_mape(y_true, y_pred):
    """
    Compute mean absolute percentage error (MAPE)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def get_mae(a, b):
    """
    Comp mean absolute error e_t = E[|a_t - b_t|]. a and b can be lists.
    Returns a vector of len = len(a) = len(b)
    """
    return np.mean(abs(np.array(a)-np.array(b)))


def get_rmse(a, b):
    """
    Comp RMSE. a and b can be lists.
    Returns a scalar.
    """
    return math.sqrt(np.mean((np.array(a)-np.array(b))**2))


def pred_xgboost(model, X_test_ex_adj_close, N, H, prev_vals, prev_mean_val, prev_std_val):
    """
    Do recursive forecasting using xgboost
    Inputs
        model              : the xgboost model
        X_test_ex_adj_close: features of the test set, excluding adj_close_scaled values
        N                  : for feature at day t, we use lags from t-1, t-2, ..., t-N as features
        H                  : forecast horizon
        prev_vals          : numpy array. If predict at time t,
                             prev_vals will contain the N unscaled values at t-1, t-2, ..., t-N
        prev_mean_val      : the mean of the unscaled values at t-1, t-2, ..., t-N
        prev_std_val       : the std deviation of the unscaled values at t-1, t-2, ..., t-N
    Outputs
        Times series of predictions. Numpy array of shape (H,). This is unscaled.
    """
    forecast = prev_vals.copy()

    for n in range(H):
        forecast_scaled = (forecast[-N:] - prev_mean_val) / prev_std_val

        # Create the features dataframe
        X = X_test_ex_adj_close[n:n + 1].copy()
        for n in range(N, 0, -1):
            X.loc[:, "adj_close_scaled_lag_" + str(n)] = forecast_scaled[-n]

        # Do prediction
        est_scaled = model.predict(X)

        # Unscale the prediction
        forecast = np.concatenate([forecast,
                                   np.array((est_scaled * prev_std_val) + prev_mean_val).reshape(1, )])

        # Comp. new mean and std
        prev_mean_val = np.mean(forecast[-N:])
        prev_std_val = np.std(forecast[-N:])

    return forecast[-H:]


def train_pred_eval_model(X_train_scaled,
                          y_train_scaled,
                          X_test_ex_adj_close,
                          y_test,
                          N,
                          H,
                          prev_vals,
                          prev_mean_val,
                          prev_std_val,
                          seed=100,
                          n_estimators=100,
                          max_depth=3,
                          learning_rate=0.1,
                          min_child_weight=1,
                          subsample=1,
                          colsample_bytree=1,
                          colsample_bylevel=1,
                          gamma=0):
    '''
      Train model, do prediction, scale back to original range and do evaluation
      Use XGBoost here.
      Inputs
          X_train_scaled     : features for training. Scaled to have mean 0 and variance 1
          y_train_scaled     : target for training. Scaled to have mean 0 and variance 1
          X_test_ex_adj_close: features of the test set, excluding adj_close_scaled values
          y_test             : target for test. Actual values, not scaled.
          N                  : for feature at day t, we use lags from t-1, t-2, ..., t-N as features
          H                  : forecast horizon
          prev_vals          : numpy array. If predict at time t,
                               prev_vals will contain the N unscaled values at t-1, t-2, ..., t-N
          prev_mean_val      : the mean of the unscaled values at t-1, t-2, ..., t-N
          prev_std_val       : the std deviation of the unscaled values at t-1, t-2, ..., t-N
          seed               : model seed
          n_estimators       : number of boosted trees to fit
          max_depth          : maximum tree depth for base learners
          learning_rate      : boosting learning rate (xgb’s “eta”)
          min_child_weight   : minimum sum of instance weight(hessian) needed in a child
          subsample          : subsample ratio of the training instance
          colsample_bytree   : subsample ratio of columns when constructing each tree
          colsample_bylevel  : subsample ratio of columns for each split, in each level
          gamma              :
      Outputs
          rmse               : root mean square error of y_test and est
          mape               : mean absolute percentage error of y_test and est
          mae                : mean absolute error of y_test and est
          est                : predicted values. Same length as y_test
      '''
    model_seed = 100
    model = XGBRegressor(objective='reg:squarederror',
                         seed=model_seed,
                         n_estimators=n_estimators,
                         max_depth=max_depth,
                         learning_rate=learning_rate,
                         min_child_weight=min_child_weight,
                         subsample=subsample,
                         colsample_bytree=colsample_bytree,
                         colsample_bylevel=colsample_bylevel,
                         gamma=gamma)

    # Train the model
    model.fit(X_train_scaled, y_train_scaled)

    # Get predicted labels and scale back to original range
    est = pred_xgboost(model, X_test_ex_adj_close, N, H, prev_vals, prev_mean_val, prev_std_val)

    # Calculate RMSE, MAPE, MAE
    rmse = get_rmse(y_test, est)
    mape = get_mape(y_test, est)
    mae = get_mae(y_test, est)

    return rmse, mape, mae, est, model.feature_importances_


def do_scaling(df, N):
    """
    Do scaling for the adj_close and lag cols
    """
    df[df.columns[0]] = (df['adj_close'] - df['adj_close_mean']) / df['adj_close_std']
    for n in range(N, 0, -1):
        df.loc[:, 'adj_close_scaled_lag_' + str(n)] = \
            (df['adj_close_lag_' + str(n)] - df['adj_close_mean']) / df['adj_close_std']

        # Remove adj_close_lag column which we don't need anymore
        df.drop(['adj_close_lag_' + str(n)], axis=1, inplace=True)

    return df


def get_error_metrics(df, train_size, N, H, seed=100, n_estimators=100, max_depth=3, learning_rate=0.1,
                      min_child_weight=1, subsample=1, colsample_bytree=1, colsample_bylevel=1, gamma=0):
    """
    Given a series consisting of both train+validation, do predictions of forecast horizon H on the validation set,
    at H/2 intervals.
    Inputs
        df                 : train + val dataframe. len(df) = train_size + val_size
        train_size         : size of train set
        N                  : for feature at day t, we use lags from t-1, t-2, ..., t-N as features
        H                  : forecast horizon
        seed               : model seed
        n_estimators       : number of boosted trees to fit
        max_depth          : maximum tree depth for base learners
        learning_rate      : boosting learning rate (xgb’s “eta”)
        min_child_weight   : minimum sum of instance weight(hessian) needed in a child
        subsample          : subsample ratio of the training instance
        colsample_bytree   : subsample ratio of columns when constructing each tree
        colsample_bylevel  : subsample ratio of columns for each split, in each level
        gamma              :
    Outputs
        mean of rmse, mean of mape, mean of mae, dictionary of predictions
    """
    rmse_list = []  # root mean square error
    mape_list = []  # mean absolute percentage error
    mae_list = []  # mean absolute error
    preds_dict = {}

    # Do scaling
    # df = do_scaling(df, N)

    # Get list of features
    features = list(set(df.columns) - {df.columns[0]})

    for i in range(train_size, len(df) - H + 1, int(H / 2)):
        # Split into train and test
        train = df[i - train_size:i].copy()
        test = df[i:i + H].copy()

        # Drop the NaNs in train
        train.dropna(axis=0, how='any', inplace=True)

        # Split into X and y
        X_train_scaled = train[features]
        y_train_scaled = train[df.columns[0]]
        X_test_ex_adj_close = test[features]
        y_test = test['adj_close']
        prev_vals = train[-N:]['adj_close'].to_numpy()
        prev_mean_val = test.iloc[0]['adj_close_mean']
        prev_std_val = test.iloc[0]['adj_close_std']

    rmse, mape, mae, est, _ = train_pred_eval_model(X_train_scaled,
                                                    y_train_scaled,
                                                    X_test_ex_adj_close,
                                                    y_test,
                                                    N,
                                                    H,
                                                    prev_vals,
                                                    prev_mean_val,
                                                    prev_std_val,
                                                    seed=seed,
                                                    n_estimators=n_estimators,
                                                    max_depth=max_depth,
                                                    learning_rate=learning_rate,
                                                    min_child_weight=min_child_weight,
                                                    subsample=subsample,
                                                    colsample_bytree=colsample_bytree,
                                                    colsample_bylevel=colsample_bylevel,
                                                    gamma=gamma)
    #         print("N = " + str(N) + ", i = " + str(i) + ", rmse = " + str(rmse) + ", mape = " + str(mape) + ", mae = " + str(mae))

    rmse_list.append(rmse)
    mape_list.append(mape)
    mae_list.append(mae)
    preds_dict[i] = est

    return np.mean(rmse_list), np.mean(mape_list), np.mean(mae_list), preds_dict



def xgboost_aux():
    import time
    from _collections import defaultdict

    param_label = 'n_estimators'
    param_list = range(1, 61, 2)

    param2_label = 'max_depth'
    param2_list = [2, 3, 4, 5, 6, 7, 8, 9]

    error_rate = defaultdict(list)

    tic = time.time()
    for param in tqdm_notebook(param_list):
        for param2 in param2_list:
            rmse_mean, mape_mean, mae_mean, _ = get_error_metrics(train_val,
                                                                  train_size,
                                                                  N_opt,
                                                                  H,
                                                                  seed=model_seed,
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
            error_rate['rmse'].append(rmse_mean)
            error_rate['mape'].append(mape_mean)
            error_rate['mae'].append(mae_mean)

    error_rate = pd.DataFrame(error_rate)
    toc = time.time()
    print("Minutes taken = {0:.2f}".format((toc - tic) / 60.0))

    error_rate