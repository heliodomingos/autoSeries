from feature_engineering_module.exponential_smoothing import *
from feature_engineering_module.moving_average import *


def smoothing_operators(time_series, series_name, alfas, target_variable):
    exponential_smoothing(time_series, alfas, target_variable)
    exponential_smoothing(time_series, 0.5, target_variable)
    exponential_smoothing(time_series, 0.8, target_variable)
    moving_average(time_series, 5, target_variable, 'report/' + series_name + '/moving_average_5.png')
    moving_average(time_series, 8, target_variable, 'report/' + series_name + '/moving_average_8.png')
    moving_average(time_series, 10, target_variable, 'report/' + series_name + '/moving_average_10.png')
    moving_average(time_series, 20, target_variable, 'report/' + series_name + '/moving_average_20.png')

    import pywt

    a, b = pywt.dwt(time_series['close'].to_numpy(), 'db2')

    a = np.concatenate((np.zeros(len(time_series) - len(a)), a))
    b = np.concatenate((np.zeros(len(time_series) - len(b)), b))
    time_series['db2_a'] = a.tolist()
    time_series['db2_c'] = b.tolist()

    return {'smoothing_operators': time_series}
