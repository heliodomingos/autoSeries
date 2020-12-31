import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import autocorrelation_plot


def test_stationary(time_series):
    """
    :param df: pandas dataframe time series with data value as index and target variable
    :return: boolean parameter - true if df is stationary false otherwise
    Testing using the Dickey-Fuller test. If p > 0, the process is not stationary (false).
    Otherwise, p = 0, the null hypothesis is rejected, and the process is considered to be stationary(true).
    """
    result = adfuller(time_series)
    # print('ADF Statistic: {}'.format(result[0]))
    # print('p-value: {}'.format(result[1]))
    # print('Critical Values:')
    # for key, value in result[4].items():
        # print('\t{}: {}'.format(key, value))
    from pmdarima.arima import ADFTest
    adf_test = ADFTest(alpha=0.05)
    should_diff_bool = adf_test.should_diff(time_series)[1]
    return should_diff_bool #result[1] < 0.05


def transform_to_stationary(time_series):
    diff_order = 0

    return (time_series, diff_order)


def trend(time_series):
    return 0


def seasonality(time_series):
    return 0


def autocor_plot(time_series):
    d = autocorrelation_plot(time_series.iloc[:, 0])

    return 0