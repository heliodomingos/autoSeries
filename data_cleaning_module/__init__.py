from data_cleaning_module.data_cleaning_methods import *
import streamlit as st


# @st.cache(allow_output_mutation=True)
def cleaning_data(time_series):
    time_series = interpolate_missing_values(time_series)
    return time_series
