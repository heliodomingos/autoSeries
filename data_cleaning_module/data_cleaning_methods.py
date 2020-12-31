def remove_duplicates(time_series):
    return time_series.drop_duplicates()


def interpolate_missing_values(time_series):
    if time_series.isnull().values.any():
        #time_series = time_series.interpolate(method='nearest', inplace=True)
        time_series.fillna(method='backfill')
        time_series.fillna(method='ffill')
    return time_series
