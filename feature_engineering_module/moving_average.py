from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from error_functions_definition import *


def moving_average(time_series, window, target_variable, figure_path, plot_intervals=False, scale=1.96):

    rolling_mean = time_series[target_variable].rolling(window=window).mean()

    plt.figure(figsize=(40, 8))
    plt.title('Moving average\n window size = {}'.format(window))
    plt.plot(rolling_mean, 'g', label='Rolling mean trend')

    # Plot confidence intervals for smoothed values
    if plot_intervals:
        mae = mean_absolute_error(time_series[window:], rolling_mean[window:])
        deviation = np.std(time_series[window:] - rolling_mean[window:])
        lower_bound = rolling_mean - (mae + scale * deviation)
        upper_bound = rolling_mean + (mae + scale * deviation)
        plt.plot(upper_bound, 'r--', label='Upper bound / Lower bound')
        plt.plot(lower_bound, 'r--')

    plt.plot(time_series[window:], label='Actual values')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(figure_path)
    time_series['moving_average_{}'.format(window)] = rolling_mean
    return time_series

