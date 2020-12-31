def raw_to_second(series, aggregate_function, target_variable):
    series['mean_by_second'] = series[target_variable].resample('S').mean()
    series['sum_by_second'] = series[target_variable].resample('S').sum()
    series['min_by_second'] = series[target_variable].resample('S').min()
    series['max_by_second'] = series[target_variable].resample('S').max()
    return series


def raw_to_minute(series, target_variable):
    series['mean_by_minute'] = series[target_variable].resample('T').mean()
    series['sum_by_minute'] = series[target_variable].resample('T').sum()
    series['min_by_minute'] = series[target_variable].resample('T').min()
    series['max_by_minute'] = series[target_variable].resample('T').max()
    return series#.resample('T').mean()


def raw_to_hour(series, target_variable):
    series['mean_by_hour'] = series[target_variable].resample('H').mean()
    series['sum_by_hour'] = series[target_variable].resample('H').sum()
    series['min_by_hour'] = series[target_variable].resample('H').min()
    series['max_by_hour'] = series[target_variable].resample('H').max()
    return series


def raw_to_day(series, target_variable):
    series['mean_by_day'] = series[target_variable].resample('D').mean()
    series['sum_by_day'] = series[target_variable].resample('D').sum()
    series['min_by_day'] = series[target_variable].resample('D').min()
    series['max_by_day'] = series[target_variable].resample('D').max()

    return series

def raw_to_week(series, target_variable):
    series['mean_by_week'] = series.groupby(series.index.week)['close'].transform('mean')
    series['sum_by_week'] = series.groupby(series.index.week)['close'].transform('sum')
    series['min_by_week'] = series.groupby(series.index.week)['close'].transform('min')
    series['max_by_week'] = series.groupby(series.index.week)['close'].transform('max')
    series['median_by_week'] = series.groupby(series.index.week)['close'].transform('median')
    return series

def raw_to_month(series, target_variable):
    series['mean_by_month'] = series.groupby(series.index.month)['close'].transform('mean')
    series['sum_by_month'] = series.groupby(series.index.month)['close'].transform('sum')
    series['min_by_month'] = series.groupby(series.index.month)['close'].transform('min')
    series['max_by_month'] = series.groupby(series.index.month)['close'].transform('max')
    series['median_by_month'] = series.groupby(series.index.month)['close'].transform('median')
    return series


def raw_to_year(series, target_variable):
    series['mean_by_year'] = series.groupby(series.index.year)['close'].transform('mean')
    series['sum_by_year'] = series.groupby(series.index.year)['close'].transform('sum')
    series['max_by_year'] = series.groupby(series.index.year)['close'].transform('max')
    series['min_by_year'] = series.groupby(series.index.year)['close'].transform('min')
    series['median_by_year'] = series.groupby(series.index.year)['close'].transform('median')
    return series


def derivative(series, target_variable):
    index = series.index
    import pandas
    if isinstance(index, pandas.DatetimeIndex):
        den = index.to_series(keep_tz=True).diff().dt.total_seconds()
    else:
        den = index.to_series().diff()
    num = series[target_variable].diff()
    series['derivative'] = num.div(den, axis=0).multiply(10000)
    return series

def derivativev2(series, target_variable):
    index = series.index
    import pandas
    if isinstance(index, pandas.DatetimeIndex):
        den = index.to_series(keep_tz=True).diff().dt.total_seconds()
    else:
        den = index.to_series().diff()
    num = series[target_variable].diff()
    return num.div(den, axis=0).multiply(10000)

def differencing(series, target_variable):
    series['difference'] = series[target_variable].diff(periods=1)

    return series



def aggregate_composition_op(time_series, feature):

    oo = time_series.columns[1:]
    for target_variable in oo:
        for e in feature:
            if e == "derivative":
                time_series[target_variable + 'XXderivative'] = derivativev2(time_series, target_variable)
            elif e == "differencing":
                time_series[target_variable + 'XXdifferencing'] = time_series.diff()
            elif e == 'up_down':
                time_series[target_variable + 'XXup_down'] = time_series[target_variable].diff()
                std = time_series[target_variable + 'XXup_down'].std()
                time_series[target_variable + 'XXup_down'] = (time_series[target_variable + 'XXup_down'] < 0).astype(int)
            elif e == 'big_down':
                std = time_series[target_variable].diff().std()
                time_series[target_variable + 'XXbig_down'] = time_series[target_variable].diff()
                time_series[target_variable + 'XXbig_down'] = (time_series[target_variable + 'XXbig_down'] < -std).astype(int)
            elif e == "big_up":
                std = time_series[target_variable].diff().std()
                time_series[target_variable + 'XXbig_up'] = time_series[target_variable].diff()
                time_series[target_variable + 'XXbig_up'] = (time_series[target_variable + 'XXbig_up'] > std).astype(int)

    time_series = time_series.drop(oo, axis=1, inplace=True)

    return time_series


def aggregate_operators(time_series, time_granularities):
    f_granularity = [raw_to_week, raw_to_month, raw_to_year]
    extra = [derivative, differencing]
    x = {}
    target_variable = time_series.columns[0]
    print('target::::', target_variable)
    print('time key: ', time_series.columns)
    for i in range(1, 4):
        if time_granularities[-i]:
            x[f_granularity[-i].__name__] = f_granularity[-i](time_series, target_variable)
            time_series = f_granularity[-i](time_series, target_variable)
    print('time key: ', time_series.columns)
    for e in extra:
        x[e.__name__] = e(time_series, target_variable)


    time_series['up_down'] = time_series[time_series.columns[0]].diff()
    std = time_series['up_down'].std()
    time_series['up_down'] = (time_series['up_down'] < 0).astype(int)

    time_series['big_down'] = time_series[time_series.columns[0]].diff()
    time_series['big_down'] = (time_series['big_down'] < -std).astype(int)

    time_series['big_up'] = time_series[time_series.columns[0]].diff()
    time_series['big_up'] = (time_series['big_up'] > std).astype(int)

    return time_series

def only_time(time_series):
    #time_series['hour'] = time_series.index.hour
    time_series['dayofweek'] = time_series.index.dayofweek
    time_series['quarter'] = time_series.index.quarter
    time_series['month'] = time_series.index.month
    time_series['year'] = time_series.index.year
    time_series['dayofyear'] = time_series.index.dayofyear
    time_series['dayofmonth'] = time_series.index.day
    time_series['weekofyear'] = time_series.index.weekofyear
    time_series["weekday"] = time_series.index.weekday
    time_series["weekday"] = (time_series["weekday"] < 5).astype(int)


    return time_series