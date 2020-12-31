import matplotlib.pyplot as plt
import pandas as pd


def recreate_df(dict_of_Series, time):
    for key in dict_of_Series.keys():
        dict_of_Series[key] = pd.DataFrame(dict_of_Series[key], time)
    return dict_of_Series


def exponential_smoothing(time_series, alpha, target_variable):
    if alpha is None:
        alpha = 0.3

    series = time_series[target_variable]
    result = [series[0]]  # first value is same as series
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n - 1])
    time_series["expo_smooth_"+str(alpha)] = result
    return time_series


def plot_exponential_smoothing(series, series_name, alphas, target_variable):
    series = getattr(series, target_variable)
    plt.figure(figsize=(17, 8))
    exponential_data = {}
    for alpha in alphas:
        exponential_data[alpha] = exponential_smoothing(series, alpha)
        plt.plot(exponential_data[alpha], label="Alpha {}".format(alpha))
    plt.plot(series.values, "c", label="Actual")
    plt.legend(loc="best")
    plt.axis('tight')
    plt.title("Exponential Smoothing - alpha=", alphas)
    plt.grid(True)
    plt.savefig("report/" + series_name + "/Exponential_smoothing_" + alphas)
    return recreate_df(exponential_data, series.index)

