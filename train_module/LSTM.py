import streamlit as st
import numpy as np
import pandas as pd
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.callbacks import EarlyStopping
from tensorflow.python.keras import Sequential


def my_scaller(series, mean=None, std=None):

    #formula = x - media / std
    if mean is None:
        mean = np.mean(series)
    if std is None:
        std = np.std(series)

    series = np.divide(np.subtract(series, mean), std)

    return series, mean, std

def my_scaller_inverse(series, mean, std):
    print('inverse')
    # formula = x - media / std

    print(series)
    print(mean)
    print(std)
    d = series + mean #np.sum(np.array(series).astype(float), mean)
    series = d * std
    return series

def create_model(shape):
    model = Sequential()
    # Adding the first LSTM layer and some Dropout regularisation
    model.add(LSTM(units=50, return_sequences=True, input_shape=(shape, 1)))
    model.add(Dropout(0.2))
    # Adding a second LSTM layer and some Dropout regularisation
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    # Adding a third LSTM layer and some Dropout regularisation
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    # Adding a fourth LSTM layer and some Dropout regularisation
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    # Adding the output layer
    model.add(Dense(units=1))
    return model


def myLSTM(original, time_series, series_name, n_diffs, z):
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
    #time_series = time_series.diff(n_diffs).dropna().mul(10**z)
    #print(time_series)
    # https://towardsdatascience.com/lstm-time-series-forecasting-predicting-stock-prices-using-an-lstm-model-6223e9644a2f
    train_size = int(len(time_series) * 0.9)
    test_size = len(time_series) - train_size
    training_set = time_series.iloc[:train_size, :].values
    test_set = time_series.iloc[train_size:, :].values
    last_value_training = original.iloc[train_size:, :].values[4]

    # Feature Scaling
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)
    #training_set_scaled, mean, std = my_scaller(training_set)
    # Creating a data structure with 60 time-steps and 1 output
    X_train = []
    y_train = []
    lag = 5
    for i in range(lag, train_size):
        X_train.append(training_set_scaled[i - lag:i, 0])
        y_train.append(training_set_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    model = create_model(X_train.shape[1])

    # Compiling the RNN
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Fitting the RNN to the Training set
    model.fit(X_train, y_train, epochs=5, batch_size=32, callbacks=[EarlyStopping(monitor='loss', patience=10)])


    dataset_train = time_series.iloc[:train_size, :]
    dataset_test = time_series.iloc[train_size:, :]
    dataset_total = pd.concat((dataset_train, dataset_test), axis=0)
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - lag:].values
    inputs = inputs.reshape(-1, 1)
    #inputs, mean_inputs, std_inputs = my_scaller(inputs, mean, std)
    inputs = sc.inverse_transform(inputs)
    X_test = []
    for i in range(lag, test_size):
        X_test.append(inputs[i - lag:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


    predicted_stock_price = model.predict(X_test)
    #predicted_stock_price =my_scaller_inverse(predicted_stock_price, mean, std)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    predicted_stock_price = np.array(predicted_stock_price).transpose()[0]
    y = time_series[time_series.columns[0]][train_size:]
    y = np.array(y).transpose()[:-lag]
    #print(time_series.index[train_size:])
    #print(np.array(time_series.index)[train_size:])

    A = pd.DataFrame({'Actual Data': y, 'Predictions': predicted_stock_price},
  #                    'Diff': y - predicted_stock_price},
                     index=np.array(time_series.index)[train_size:-lag])

    corr = np.corrcoef(y, predicted_stock_price)[0][1]

    ###########################################
    #          Correlation                    #
    #              Plot                       #
    ###########################################
    import scipy.stats
    import matplotlib.pyplot as plt
    slope, intercept, r, p, stderr = scipy.stats.linregress(y, predicted_stock_price)
    line = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r={r:.2f}'
    st.write(line)
    fig, ax = plt.subplots()
    ax.plot(y, predicted_stock_price, linewidth=0, marker='o', label='Data points', c='DarkBlue')
    ax.plot(y, intercept + slope * y, label=line, c='orange')
    ax.set_xlabel('Actual Data')
    ax.set_ylabel('Predictions')
    ax.legend(facecolor='white')
    plt.savefig('report/' + series_name + "/baseline_regression_line.png")


    #reverse diffs
    original['n'+str(n_diffs)] = time_series


    for i in range(n_diffs):
        x, x_diff = original['close'].iloc[n_diffs-1-i], original['n'+str(n_diffs-i)].iloc[n_diffs-i:]
        original['n'+str(n_diffs-1-i)] = np.r_[x, x_diff].cumsum().astype(float)

    st.write(original)


    predicts_finais = []

    x = last_value_training[0]
    for p in predicted_stock_price:
        predicts_finais.append(x+p*10**z)
        x = x + p

    oooo = original['n0'].values[train_size + lag + n_diffs:]
    B = pd.DataFrame({'Original': oooo, 'Predictions + Diffs': predicts_finais,
                               'Diff': oooo - predicts_finais},
                              index=np.array(time_series.index)[train_size:-lag])
    sem_diffs = [mean_absolute_error(oooo, predicts_finais),
                 np.sqrt(mean_squared_error(oooo, predicts_finais)),
                 B]




    #st.write("Testing diffs")
    #st.line_chart(sem_diffs[2])
    #st.write(B)
    #st.write(sem_diffs[0])
    #st.write(sem_diffs[1])
    #st.write("--------")

    com_diffs = [mean_absolute_error(y, predicted_stock_price),
            np.sqrt(mean_squared_error(y, predicted_stock_price)),
                 np.mean(np.abs((y - predicted_stock_price) / y)) * 100,
            A,
                 cal_accuracy(y,predicted_stock_price, y_train[-1]),
                 corr]


    return com_diffs


def b_LSTM(time_series, series_name, n_diffs):
    dates = np.array(time_series.index)
    dataset = time_series.values
    # dataset = dataset.astype('float32')

    # ensure all data is float
    dataset = dataset.astype('float32')

    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

    train_dates, test_dates = dates[0:train_size], dates[train_size:len(dataset)]

    def create_dataset(dataset, look_back=1):
        X, Y = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            X.append(i)
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)

    def create_dates(dataset, look_back=1):
        X, Y = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back)]
            X.append(a)
            Y.append(dataset[i + look_back])
        return np.array(X), np.array(Y)

    look_back = 1
    X_train, Y_train = create_dataset(train, look_back)
    X_train_dates, Y_train_dates = create_dates(train_dates, look_back)
    X_test, Y_test = create_dataset(test, look_back)
    X_test_dates, Y_test_dates = create_dates(test_dates, look_back)

    # reshape input to be [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], 1, 1))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, 1))
    model = Sequential()
    model.add(LSTM(1, input_shape=(X_train.shape[0], 1)))
    model.add(Dropout(0.2))
    #model.add(LSTM(100, return_sequences=True))
    #model.add(Dropout(0.2))
    # Adding a second LSTM layer and some Dropout regularisation
    #model.add(LSTM(units=50))
    #model.add(Dropout(0.2))
    # Adding a third LSTM layer and some Dropout regularisation
    #model.add(LSTM(units=50))
    #model.add(Dropout(0.2))
    # Adding a fourth LSTM layer and some Dropout regularisation
    #model.add(LSTM(units=50))
    #model.add(Dropout(0.2))
    # Adding the output layer
    model.add(Dense(units=1))

    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(X_train, Y_train, epochs=100, batch_size=32,
                        callbacks=[EarlyStopping(monitor='loss', patience=10)], verbose=False, shuffle=False)

    model.summary()
    test_predict = model.predict(X_test)
    # invert predictions
    test_predict = scaler.inverse_transform(test_predict)
    Y_test = scaler.inverse_transform([Y_test])

    A = pd.DataFrame({'Actual Data': Y_test[0], 'Predictions': test_predict[:, 0][:test_size], 'Diff':Y_test[0]-test_predict[:, 0][:test_size]},
                     index=Y_test_dates)

    return [mean_absolute_error(Y_test[0], test_predict[:, 0]),
            np.sqrt(mean_squared_error(Y_test[0], test_predict[:, 0])),
            A]

#@st.cache(allow_output_mutation=True)
def b_LSTM_old(time_series, series_name):
    dates = np.array(time_series.index.astype(int))
    print('datassss')
    print(dates)
    batch = 2
    dataset = time_series.values
    #dataset = dataset.astype('float32')
    # ensure all data is float
    dataset = dataset.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler2 = MinMaxScaler(feature_range=(0, 1))
    #dataset = scaler.fit_transform(dataset)
    train_size = int(len(dataset) * 0.9)
    test_size = len(dataset) - train_size
    print('lstm, train: ', train_size, ' test: ', test_size)
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    train_dates, test_dates = dates[0:train_size], dates[train_size:len(dataset)]
    train = scaler.fit_transform(train)
    test = scaler.fit_transform(test)
    # reshape input to be [samples, time steps, features]
    train = np.reshape(train, (1, batch, 1))
    #test = np.reshape(test, (test.shape[0], test.shape[1], 1))
    train_dates = np.reshape(train_dates, (1, batch, 1))
    test_dates = np.reshape(test_dates, (1, batch, 1))
    model = Sequential()
    print('input shape: ', train.shape[1])
    model.add(Bidirectional(LSTM(50, input_shape=(batch, 1), return_sequences=True)))
    #model.add(Dropout(0.2))
    # Adding a second LSTM layer and some Dropout regularisation
    #model.add(LSTM(units=50, return_sequences=True))
    #model.add(Dropout(0.4))
    # Adding a third LSTM layer and some Dropout regularisation

    #model.add(LSTM(units=50, return_sequences=True))
    #model.add(Dropout(0.2))
    # Adding a fourth LSTM layer and some Dropout regularisation
    #model.add(LSTM(units=50, return_sequences=True))
    #model.add(Dropout(0.2))
    # Adding the output layer
    model.add(TimeDistributed(Dense(units=1)))

    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(train_dates, train, epochs=100, batch_size=10,
                        callbacks=[EarlyStopping(monitor='loss', patience=10)], verbose=2, shuffle=False)

    model.summary()
    test_predict = model.predict(test_dates)
    # invert predictions
    test_predict = scaler.inverse_transform(test_predict)
    #test = scaler.inverse_transform(test)
    d =np.array(time_series.index[train_size:len(dataset)])
    #print(d.shape[0])
    #d = d.reshape(d.shape[0], 1)
    test_predict = test_predict.reshape(-1, 1)
    print('actual: ', test.shape, 'prediticions', test_predict.shape)
    A = pd.DataFrame({
        'Actual Data': test[:, 0],
        'Predictions': test_predict[:, 0][:test_size],
        'Diff': test[:, 0]-test_predict[:, 0][:test_size]
    }, index=d)
    #A.set_index('Date')

                     #index=np.array(time_series.index[train_size:len(dataset)]).reshape(-1,1))
    print('predictions')
    print(test_predict[:, 0][:test_size])
    return [mean_absolute_error(test, test_predict[:, 0]),
            np.sqrt(mean_squared_error(test, test_predict[:, 0])),
            A]