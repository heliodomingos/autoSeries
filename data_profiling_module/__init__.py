from data_profiling_module.data_profiling_methods import *
from draw_data_module import *
import pandas as pd
import numpy as np

def stationarity(time_series, series_name):
    window = int(len(time_series) * .01)
    rolling_mean = time_series.ix[:, 0].rolling(window=window).mean()
    rolling_std = time_series.ix[:, 0].rolling(window=window).std()
    plt.plot(time_series.ix[:, 0], color='blue', label='Original')
    plt.plot(rolling_mean, color='red', label='Rolling Mean')
    plt.plot(rolling_std, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Rolling Standard Deviation')
    plt.savefig('report/' + series_name + '_rolling_mean.png')
    return 0


def get_variations(time_series):
    import numpy as np
    variation = time_series.copy()
    variation[1:] = (time_series[1:] / time_series[:-1].values) - 1
    variation.ix[0, :] = 0
    return variation


def get_distribution_hist(time_series, series_name, path=""):
    fig, ax = plt.subplots(figsize=(8, 8))
    #time_series = time_series[time_series["close"] > -80]
    #time_series = time_series[time_series["close"] < 80]
    time_series.hist(time_series.columns[0], ax=ax)
    ax.set_title('Data distribution')
    ax.set_xlabel('Values')
    ax.set_ylabel('Number of values')
    media = time_series.ix[:, 0].mean()
    deviation = time_series.ix[:, 0].std()
    plt.axvline(media, color='b', linestyle='dashed', linewidth=2, label='mean')
    plt.axvline(media + deviation, color='r', linestyle='dashed', linewidth=2, label='std')
    plt.axvline(media - deviation, color='r', linestyle='dashed', linewidth=2)
    plt.legend(loc='best')
    fig.savefig('report/' + series_name + '/hist' + path + '.png', bbox_inches='tight')
    plt.cla()
    return 'report/' + series_name + '/hist' + path + '.png'


def get_decomposition_fig(time_series, series_name):
    time_series.index = pd.to_datetime(time_series.index)
    decomposition = sm.tsa.seasonal_decompose(time_series, freq=52)

    fig = decomposition.plot()
    fig.savefig('report/' + series_name + "/decomposition.png")
    return 'report/' + series_name + "/decomposition.png"


def plot_autocorrelation(time_series, series_name):
    from pandas.plotting import lag_plot
    time_series = time_series.reset_index()
    time_series = time_series[time_series.columns[1]]
    plt.figure()
    lag_plot(time_series, lag=3)
    plt.title(series_name + ' - Autocorrelation plot with lag = 3')
    plt.savefig("report/" + series_name + "/autocorrelation.png")
    return "report/" + series_name + "/autocorrelation.png"


def plot_log_diff(time_series, series_name, target_variable='close'):
    fig, ax = plt.subplots(figsize=(8, 8))

    time_series['diff'] = np.log(time_series[target_variable].diff())
    time_series = time_series.reset_index()

    time_series.plot(kind='line', x='date', y='diff', ax=ax)
    plt.figure()
    plt.title(series_name + ' - Logarithm difference')
    plt.savefig("report/" + series_name + "/log_diff.png")
    return "report/" + series_name + "/log_diff.png"


def explore_data(time_series, series_name):
    plot_series(time_series, 'close', 'report/' + series_name + '/initial_data.png', series_name)
    insights = {
        'hist': get_distribution_hist(time_series, series_name),
        'initial_data_plot': 'report/' + series_name + '/initial_data.png',
        'trend': trend(time_series),
        'seasonality': seasonality(time_series.ix[:, 0]),
        'statistics': time_series.describe(),
        'decomposition_fig': get_decomposition_fig(time_series, series_name)  # ,
        # 'autocorrelation_plot': plot_autocorrelation(time_series[time_series.columns[0]].copy(), series_name)
    }

    stationarity(time_series, series_name)
    insights['isStationary'] = test_stationary(time_series.ix[:, 0])
    should_difff = test_stationary(time_series)
    i = 0
    deleted_timestamps = []
    new_time_series = time_series.copy()
    while should_difff == True: #insights['isStationary'] != False: #meaning is non-stationary
        i += 1
        new_time_series = new_time_series.diff(i).dropna()
        insights['isStationary'] = test_stationary(new_time_series.ix[:, 0])
        should_difff = test_stationary(new_time_series)

    z = 0
    if new_time_series.values.mean() < 1:
        n = str(new_time_series.values.mean()).split(".")[1]
        for j in n:
            if j == '0':
                z += 1
            else:
                break

        new_time_series = new_time_series.mul(10**z)


    should_difff = test_stationary(new_time_series)
    insights['stationaryData'] = new_time_series

    insights['z_multiple'] = z

    import pandas
    insights['variation'] = get_variations(pandas.DataFrame(time_series.iloc[:, 0].copy()))
    insights['diffs'] = i
    # plot_log_diff(time_series, series_name)
    plot_autocorrelation(time_series, series_name)
    return insights
