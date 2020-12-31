from load_data_module.load_data_methods import *
from load_data_module.timestamp_analysis import interpret_timestamp as it
import streamlit as st


# @st.cache
def load_data(time_series, filename, target_variables=None):
    file_type = filename.split(".")[-1]
    if file_type == "csv":
        time_series = load_data_from_csv(time_series, filename, target_variables)
    elif file_type == "sql":
        time_series = load_data_from_sql(time_series, filename)
    elif file_type == "xls":
        time_series = load_data_from_xls(time_series, filename)
    elif file_type == "tsv":
        time_series = load_data_from_tsv(time_series, filename)
    elif file_type == "txt":
        time_series = load_data_from_txt(time_series, filename, target_variables)
    else:
        raise ValueError("Unrecognized file type: ", file_type)
    time_series.columns = time_series.columns.str.lower()
    return time_series


def interpret_timestamp(timestamp_list):
    return it(timestamp_list)
