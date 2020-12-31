import streamlit as st
import numpy as np
import pandas as pd
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.callbacks import EarlyStopping
from tensorflow.python.keras import Sequential
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf


class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                 train_df, val_df, test_df,
                 label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)

        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32, )

        ds = ds.map(self.split_window)

        return ds

    def plot(self, model=None, plot_col='close', max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(3, 1, n + 1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time [h]')
        plt.savefig("report/plot.png")


def copiado_da_net(time_series):
    time_series['year'] = time_series.index.year
    time_series['month'] = time_series.index.month
    time_series['day'] = time_series.index.day
    column_indices = {name: i for i, name in enumerate(time_series.columns)}

    n = len(time_series)
    train_df = time_series[0:int(n * 0.7)]
    val_df = time_series[int(n * 0.7):int(n * 0.9)]
    test_df = time_series[int(n * 0.9):]

    num_features = time_series.shape[1]

    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    df_std = (time_series - train_mean) / train_std
    df_std = df_std.melt(var_name='Column', value_name='Normalized')
    plt.figure(figsize=(12, 6))
    ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
    _ = ax.set_xticklabels(time_series.keys(), rotation=90)
    plt.savefig("report/tensorflow.png")

    w1 = WindowGenerator(input_width=24, label_width=1, shift=24,
                         label_columns=['close'], test_df=test_df, val_df=val_df, train_df=train_df)

    w2 = WindowGenerator(input_width=6, label_width=1, shift=1,
                         label_columns=['close'], test_df=test_df, val_df=val_df, train_df=train_df)

    dataset = WindowGenerator.make_dataset

    train = WindowGenerator.train
    val = WindowGenerator.val
    test = WindowGenerator.test
    print(test)

    single_step_window = WindowGenerator(
        input_width=1, label_width=1, shift=1,
        label_columns=['close'], train_df=train_df, test_df=test_df, val_df=val_df)

    wide_window = WindowGenerator(
        input_width=24, label_width=24, shift=1,
        label_columns=['close'], train_df=train_df, test_df=test_df, val_df=val_df)

    lstm_model = tf.keras.models.Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        tf.keras.layers.LSTM(32, return_sequences=True),
        # Shape => [batch, time, features]
        tf.keras.layers.Dense(units=1)
    ])

    print('Input shape:', wide_window.example[0].shape)
    print('Output shape:', lstm_model(wide_window.example[0]).shape)

    MAX_EPOCHS = 2

    def compile_and_fit(model, window, patience=2):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          patience=patience,
                                                          mode='min')

        model.compile(loss=tf.losses.MeanSquaredError(),
                      optimizer=tf.optimizers.Adam(),
                      metrics=[tf.metrics.MeanAbsoluteError()])

        history = model.fit(window.train, epochs=MAX_EPOCHS,
                            validation_data=window.val,
                            callbacks=[early_stopping])
        return history

    history = compile_and_fit(lstm_model, wide_window)
    print(wide_window.val)
    val_performance = lstm_model.evaluate(wide_window.val)
    performance = lstm_model.evaluate(wide_window.test, verbose=0)

    wide_window.plot(lstm_model)

    x = np.arange(len(performance))
    width = 0.3
    metric_name = 'mean_absolute_error'
    metric_index = lstm_model.metrics_names.index('mean_absolute_error')
    #print(val_performance)
    val_mae = val_performance[metric_index]#[v[metric_index] for v in val_performance]
    test_mae = val_performance[metric_index]# [v[metric_index] for v in performance]

    plt.ylabel('mean_absolute_error [T (degC), normalized]')
    plt.bar(x - 0.17, val_mae, width, label='Validation')
    plt.bar(x + 0.17, test_mae, width, label='Test')
    plt.xticks(ticks=x, labels=performance,#.keys(),
               rotation=45)
    _ = plt.legend()
    plt.show()
    plt.savefig("report/mea.png")
    lstm_mse_error = 0
    lstm_rmse_error = 0

    # Stack three slices, the length of the total window:
    example_window = tf.stack([np.array(train_df[:w2.total_window_size]),
                               np.array(train_df[100:100 + w2.total_window_size]),
                               np.array(train_df[200:200 + w2.total_window_size])])

    example_inputs, example_labels = w2.split_window(example_window)

    print('All shapes are: (batch, time, features)')
    print(f'Window shape: {example_window.shape}')
    print(f'Inputs shape: {example_inputs.shape}')
    print(f'labels shape: {example_labels.shape}')

    return [lstm_mse_error,
            lstm_rmse_error,
            time_series]


def my_scaller(series, mean=None, std=None):
    # formula = x - media / std
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
    d = series + mean  # np.sum(np.array(series).astype(float), mean)
    series = d * std
    return series


def create_model(shape):
    model = Sequential()
    # Adding the first LSTM layer and some Dropout regularisation
    model.add(LSTM(units=15, return_sequences=True))  # , input_shape=(shape[1], shape[2])))
    model.add(Dropout(0.2))
    # Adding a second LSTM layer and some Dropout regularisation
    model.add(LSTM(units=15, return_sequences=True))
    model.add(Dropout(0.2))
    # Adding a third LSTM layer and some Dropout regularisation
    model.add(LSTM(units=15, return_sequences=True))
    model.add(Dropout(0.2))
    # Adding a fourth LSTM layer and some Dropout regularisation
    model.add(LSTM(units=15))
    model.add(Dropout(0.2))
    # Adding the output layer
    model.add(Dense(units=1))
    return model


def call_kaggle(df):
    train_data = df[:len(df) - 12]
    test_data = df[len(df) - 12:]

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    # scaler.fit(train_data)
    scaled_train_data = scaler.fit_transform(train_data)
    scaled_test_data = scaler.transform(test_data)
    from keras.preprocessing.sequence import TimeseriesGenerator

    n_input = 12
    n_features = 1
    generator = TimeseriesGenerator(scaled_train_data, scaled_train_data, length=n_input, batch_size=1)
    print(generator)
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM

    lstm_model = Sequential()
    lstm_model.add(LSTM(50, activation='relu', input_shape=(n_input, n_features)))

    lstm_model.add(Dense(1))
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit_generator(generator, epochs=2)
    losses_lstm = lstm_model.history.history['loss']

    plt.figure(figsize=(12, 4))
    plt.xticks(np.arange(0, 21, 1))
    plt.plot(range(len(losses_lstm)), losses_lstm);
    plt.savefig("report/kaggle.png")
    lstm_predictions_scaled = list()

    batch = scaled_train_data[-n_input:]
    current_batch = batch.reshape((1, n_input, n_features))

    for i in range(len(test_data)):
        lstm_pred = lstm_model.predict(current_batch)[0]
        lstm_predictions_scaled.append(lstm_pred)
        current_batch = np.append(current_batch[:, 1:, :], [[lstm_pred]], axis=1)

    lstm_predictions = scaler.inverse_transform(lstm_predictions_scaled)
    st.write(lstm_predictions)
    test_data['LSTM_Predictions'] = lstm_predictions
    st.line_chart(test_data)
    st.write(test_data)
    from sklearn.metrics import mean_squared_error
    from statsmodels.tools.eval_measures import rmse

    lstm_rmse_error = rmse(test_data['close'], test_data["LSTM_Predictions"])
    lstm_mse_error = lstm_rmse_error ** 2
    mean_value = df['close'].mean()
    print(f'MSE Error: {lstm_mse_error}\nRMSE Error: {lstm_rmse_error}\nMean: {mean_value}')

    st.write(f'MSE Error: {lstm_mse_error}\nRMSE Error: {lstm_rmse_error}\nMean: {mean_value}')

    return [lstm_mse_error,
            lstm_rmse_error,
            test_data]


def myLSTM(original, time_series, series_name, n_diffs, z):
    #return copiado_da_net(time_series)
    # return call_kaggle(time_series)
    print("#################################")
    print("#          LSTM                 #")
    print("#################################")
    time_series['year'] = time_series.index.year
    time_series['month'] = time_series.index.month
    time_series['day'] = time_series.index.day

    print("")
    train_size = int(len(time_series) * 0.9)
    test_size = len(time_series) - train_size
    training_set = time_series.iloc[:train_size, :].values
    test_set = time_series.iloc[train_size:, :].values
    last_value_training = original.iloc[train_size:, :].values[4]
    # Feature Scaling
    # sc = MinMaxScaler(feature_range=(0, 1))
    # training_set_scaled = sc.fit_transform(training_set)
    # training_set_scaled, mean, std = my_scaller(training_set)
    # Creating a data structure with 60 time-steps and 1 output
    X_train = []
    y_train = []
    lag = 5

    for i in range(lag, train_size):
        r = [training_set[i - lag:i - 1, :]]
        # r.append(training_set[i, 1:])
        # r.append(training_set[i - lag:i, 3])
        print(r)
        X_train.append(r)
        # .flatten())
        y_train.append(np.array(training_set[i, 0]))
    X_train, y_train = np.array(X_train), np.array(y_train)
    print("Xtrain 0", X_train[0].shape)
    model = create_model(X_train.shape)

    # Compiling the RNN
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Fitting the RNN to the Training set
    model.fit(X_train, y_train, epochs=10, batch_size=32, callbacks=[EarlyStopping(monitor='loss', patience=10)])

    dataset_train = time_series.iloc[:train_size, :]
    dataset_test = time_series.iloc[train_size:, :]
    dataset_total = pd.concat((dataset_train, dataset_test), axis=0)
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - lag:].values
    # inputs = inputs.reshape(-1, 1)
    # inputs, mean_inputs, std_inputs = my_scaller(inputs, mean, std)
    # inputs = sc.inverse_transform(inputs)
    X_test = []
    values = time_series.values
    '''for i in range(train_size, test_size + train_size):
        print("\n\n#########################")
        print(i)
        r = [training_set[i - lag:i, 1]]
        r.append(values[i - lag:i, 2])
        r.append(values[i - lag:i, 3])
        r = np.array(r)
        #r = np.reshape(r, (r.shape[0], 1, 1))
        print(r)
        print(r.shape)
        X_test.append(r)
        # .flatten())
        #y_train.append(np.array(values[i, 0]))
        y_hat = model.predict(r)
        print(y_hat)'''
    X_test = []
    y_train = []
    lag = 5
    total_set = time_series.iloc[:, :].values
    for i in range(train_size, test_size + train_size):
        r = [total_set[i - lag:i, 1]]
        r.append(total_set[i - lag:i, 2])
        r.append(total_set[i - lag:i, 3])
        X_test.append(r)
        # .flatten())
        # y_train.append(np.array(training_set[i, 0]))
    # X_train, y_train = np.array(X_train), np.array(y_train)

    X_test = np.array(X_test)
    st.write("pppppp", model.predict(np.array(X_test[0])))
    predicted_stock_price = model.predict(X_test)

    # predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    predicted_stock_price = np.array(predicted_stock_price).transpose()[0]
    y = time_series[time_series.columns[0]][train_size:]
    y = np.array(y).transpose()

    A = pd.DataFrame({'Actual Data': y, 'Predictions': predicted_stock_price,
                      'Diff': y - predicted_stock_price},
                     index=np.array(time_series.index)[train_size:])

    # reverse diffs
    # original['n'+str(n_diffs)] = time_series

    # for i in range(n_diffs):
    #    x, x_diff = original['close'].iloc[n_diffs-1-i], original['n'+str(n_diffs-i)].iloc[n_diffs-i:]
    #    original['n'+str(n_diffs-1-i)] = np.r_[x, x_diff].cumsum().astype(float)

    # st.write(original)

    predicts_finais = []

    x = last_value_training[0]
    for p in predicted_stock_price:
        predicts_finais.append(x + p * 10 ** z)
        x = x + p

    # oooo = original['n0'].values[train_size + lag + n_diffs:]
    # B = pd.DataFrame({'Original': oooo, 'Predictions + Diffs': predicts_finais,
    #                           'Diff': oooo - predicts_finais},
    #                          index=np.array(time_series.index)[train_size:-lag])
    # sem_diffs = [mean_absolute_error(oooo, predicts_finais),
    #             np.sqrt(mean_squared_error(oooo, predicts_finais)),
    #             B]

    # st.write("Testing diffs")
    # st.line_chart(sem_diffs[2])
    # st.write(B)
    # st.write(sem_diffs[0])
    # st.write(sem_diffs[1])
    # st.write("--------")

    com_diffs = [mean_absolute_error(y, predicted_stock_price),
                 np.sqrt(mean_squared_error(y, predicted_stock_price)),
                 A]

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
    # model.add(LSTM(100, return_sequences=True))
    # model.add(Dropout(0.2))
    # Adding a second LSTM layer and some Dropout regularisation
    # model.add(LSTM(units=50))
    # model.add(Dropout(0.2))
    # Adding a third LSTM layer and some Dropout regularisation
    # model.add(LSTM(units=50))
    # model.add(Dropout(0.2))
    # Adding a fourth LSTM layer and some Dropout regularisation
    # model.add(LSTM(units=50))
    # model.add(Dropout(0.2))
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

    A = pd.DataFrame({'Actual Data': Y_test[0], 'Predictions': test_predict[:, 0][:test_size],
                      'Diff': Y_test[0] - test_predict[:, 0][:test_size]},
                     index=Y_test_dates)

    return [mean_absolute_error(Y_test[0], test_predict[:, 0]),
            np.sqrt(mean_squared_error(Y_test[0], test_predict[:, 0])),
            A]


# @st.cache(allow_output_mutation=True)
def b_LSTM_old(time_series, series_name):
    dates = np.array(time_series.index.astype(int))
    print('datassss')
    print(dates)
    batch = 2
    dataset = time_series.values
    # dataset = dataset.astype('float32')
    # ensure all data is float
    dataset = dataset.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler2 = MinMaxScaler(feature_range=(0, 1))
    # dataset = scaler.fit_transform(dataset)
    train_size = int(len(dataset) * 0.9)
    test_size = len(dataset) - train_size
    print('lstm, train: ', train_size, ' test: ', test_size)
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    train_dates, test_dates = dates[0:train_size], dates[train_size:len(dataset)]
    train = scaler.fit_transform(train)
    test = scaler.fit_transform(test)
    # reshape input to be [samples, time steps, features]
    train = np.reshape(train, (1, batch, 1))
    # test = np.reshape(test, (test.shape[0], test.shape[1], 1))
    train_dates = np.reshape(train_dates, (1, batch, 1))
    test_dates = np.reshape(test_dates, (1, batch, 1))
    model = Sequential()
    print('input shape: ', train.shape[1])
    model.add(Bidirectional(LSTM(50, input_shape=(batch, 1), return_sequences=True)))
    # model.add(Dropout(0.2))
    # Adding a second LSTM layer and some Dropout regularisation
    # model.add(LSTM(units=50, return_sequences=True))
    # model.add(Dropout(0.4))
    # Adding a third LSTM layer and some Dropout regularisation

    # model.add(LSTM(units=50, return_sequences=True))
    # model.add(Dropout(0.2))
    # Adding a fourth LSTM layer and some Dropout regularisation
    # model.add(LSTM(units=50, return_sequences=True))
    # model.add(Dropout(0.2))
    # Adding the output layer
    model.add(TimeDistributed(Dense(units=1)))

    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(train_dates, train, epochs=100, batch_size=10,
              callbacks=[EarlyStopping(monitor='loss', patience=10)], verbose=2, shuffle=False)

    model.summary()
    test_predict = model.predict(test_dates)
    # invert predictions
    test_predict = scaler.inverse_transform(test_predict)
    # test = scaler.inverse_transform(test)
    d = np.array(time_series.index[train_size:len(dataset)])
    # print(d.shape[0])
    # d = d.reshape(d.shape[0], 1)
    test_predict = test_predict.reshape(-1, 1)
    print('actual: ', test.shape, 'prediticions', test_predict.shape)
    A = pd.DataFrame({
        'Actual Data': test[:, 0],
        'Predictions': test_predict[:, 0][:test_size],
        'Diff': test[:, 0] - test_predict[:, 0][:test_size]
    }, index=d)
    # A.set_index('Date')

    # index=np.array(time_series.index[train_size:len(dataset)]).reshape(-1,1))
    print('predictions')
    print(test_predict[:, 0][:test_size])
    return [mean_absolute_error(test, test_predict[:, 0]),
            np.sqrt(mean_squared_error(test, test_predict[:, 0])),
            A]
