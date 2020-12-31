from feature_engineering_module.aggregation_operators import *
from feature_engineering_module.smoothing_operators import *
from feature_engineering_module.tsfresh_gen import *


# Timestamp Operators: Aggregate by date
def aggregate_timestamp(time_series, time_granularities):
    return aggregate_operators(time_series, time_granularities)

def aggregate_composition(time_series, feature):
    return aggregate_composition_op(time_series, feature)

def smoothing_timestamp(time_series, series_name, alfas, target_variable):
    return smoothing_operators(time_series, series_name, alfas, target_variable)


def use_tsfresh(time_series):
    return tsfresh_gen(time_series)


def weekday(time_series):
    time_series["weekday"] = time_series.apply(lambda row: row["date"].weekday(), axis=1)
    time_series["weekday"] = (time_series["weekday"] < 5).astype(int)
    return time_series


def up_down(time_series):
    time_series['up_down'] = time_series.diff()
    time_series['up_down'] = (time_series['up_down'] < 0).astype(int)
    return time_series

def dfs(time_series):
    import featuretools as ft
    return time_series