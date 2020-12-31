from tsfresh import extract_features
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute, make_forecasting_frame
from tsfresh import extract_relevant_features


def tsfresh_gen(time_series):
    print("tsfresh")
    time_series = time_series[[time_series.columns[0]]].copy()

    #time_series = time_series.iloc[:, 0].copy()
    # print(time_series[time_series.isnull().any(axis=1)])
    #time_series['date'] = time_series.index
    time_series.reset_index(level=0, inplace=True)
    time_series.reset_index(level=0, inplace=True)


    from tsfresh import extract_features
    extracted_features = extract_features(time_series, column_id='close', column_sort='date')
    #print(extracted_features)
    impute(extracted_features)
    #features_filtered = select_features(extracted_features, 'close')
    #features_filtered_direct = extract_relevant_features(time_series, 'close',
     #                                                    column_id=time_series.columns[0], column_sort='date')

    #print('Features')
    #print(features_filtered_direct)
    #print(time_series)
    import pandas as pd
    return pd.concat([time_series, extracted_features], axis=1, sort=False)