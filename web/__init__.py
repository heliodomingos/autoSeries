import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from PIL import Image


class WebPage:
    def __init__(self):
        self.f_progress = 0
        self.f_state = ''
        self.f_state_text = ''
        st.write("""
        # Framework for automated time series analysis
        """)

    def create_forecast_section(self):
        import os
        st.markdown('<style>' + open(os.path.abspath('./web/icon.css')).read() + '</style>', unsafe_allow_html=True)
        # st.markdown('<i class="material-icons">face</i>', unsafe_allow_html=True)

        st.header('Forecast')
        self.f_progress = st.progress(0)
        self.f_state_text = 'Loading Data...'
        self.f_state = st.text(self.f_state_text)
        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader('Data')


    def f_load_data(self, data, key):
        # st.write('The input data should have at least one date column.')+

        string = str("<h2 style='text-align: center; color: #08568B;'><b>" + key + "</b></h2>")
        st.markdown(string, unsafe_allow_html=True)
        st.write(data[key]['raw_data'])
        for k in data[key]['insights'].keys():
            if k == 'initial_data_plot':
                # image = Image.open(data[key]['insights']['initial_data_plot'])
                # st.image(image, caption=key+' plot', use_column_width=True)
                st.line_chart(data[key]['raw_data'].iloc[:, 0])
            elif k == 'variation':
                st.write('Variation from first timestamp')
                st.line_chart(data[key]['insights']['variation'])
            elif k == 'hist':
                st.image(data[key]['insights']['hist'])
            elif k == 'decomposition_fig':
                st.image(data[key]['insights']['decomposition_fig'])
            elif k == 'autocorrelation_plot':
                st.image(data[key]['insights']['autocorrelation_plot'])
            else:
                st.write(k, ': ', data[key]['insights'][k])
        self.f_progress.progress(10)
        self.f_state_text = 'Data Loaded'
        self.f_state.text(self.f_state_text + ' -> Cleaning Data...')

    def f_data_cleaning(self):
        self.f_state_text += ' -> Data Cleaned'
        self.f_progress.progress(20)
        self.f_state.text(self.f_state_text + ' -> Creating Baseline...')

    def f_create_baseline(self):
        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader('Baseline')
        st.write(
            'Our baseline consists in applying a Long Short-Term Memory Neural Network (LSTM) to raw data. The results '
            'are the following:')

    def f_add_baseline_entry(self, data, key):

        string = str("<h2 style='text-align: center; color: #08568B;'><b>" + key + "</b></h2>")
        st.markdown(string, unsafe_allow_html=True)

        st.write('Test Mean Absolute Error:', data[key]['baseline']['Test Mean Absolute Error'])
        st.write('Test Root Mean Squared Error:', data[key]['baseline']['Test Root Mean Squared Error'])
        st.write(data[key]['baseline']['Predictions'])
        st.line_chart(data[key]['baseline']['Predictions'])
        #st.image(data[key]['baseline']['diff_plot'])
        st.write('Time: ', data[key]['baseline']['time(s)'], ' seconds')

    def f_add_feature_entry(self, data):
        for key in data.keys():
            string = str("<h2 style='text-align: center; color: #08568B;'><b>" + key + "</b></h2>")
            st.markdown(string, unsafe_allow_html=True)
            st.write('Features', data[key]['raw_data'])
            st.write('Test Mean Absolute Error:', data[key]['baseline']['Test Mean Absolute Error'])
            st.write('Test Root Mean Squared Error:', data[key]['baseline']['Test Root Mean Squared Error'])
            st.write(data[key]['baseline']['Predictions'])
            st.line_chart(data[key]['baseline']['Predictions'])
            st.write('Time: ', data[key]['baseline']['time(s)'], ' seconds')

    def f_update_baseline_status(self):
        st.markdown("<hr>", unsafe_allow_html=True)
        self.f_state_text += ' -> Baseline'
        self.f_progress.progress(40)
        self.f_state.text(self.f_state_text + ' -> Generating Features...')


    def f_update_feature_status(self):
        self.f_state_text += ' -> Features Generated'
        self.f_progress.progress(60)
        self.f_state.text(self.f_state_text + ' -> Training Models...')


    def f_feature_engineering(self):
        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader('Feature Engineering')


    def check_better_mae_baseline(self, data, key, index):
        try:
            b = data[key]['baseline']['Test Mean Absolute Error']
        except:
            return
        if b > data[key][index]['Test Mean Absolute Error']:
            string = str("<p style='color: #29C684;'> <b>This is better than baseline.</b></p>")
            st.markdown(string, unsafe_allow_html=True)
        else:
            string = str("<p style='color: #C63329;'> <b>This is worst than baseline.</b></p>")
            st.markdown(string, unsafe_allow_html=True)

    def check_better_rmse_baseline(self, data, key, index):
        try:
            b = data[key]['baseline']['Test Root Mean Squared Error']
        except:
            return
        if b > data[key][index]['Test Root Mean Squared Error']:
            string = str("<p style='color: #29C684;'> <b>This is better than baseline.</b></p>")
            st.markdown(string, unsafe_allow_html=True)
        else:
            string = str("<p style='color: #C63329;'> <b>This is worst than baseline.</b></p>")
            st.markdown(string, unsafe_allow_html=True)

    def check_better_mape_baseline(self, data, key, index):
        try:
            b = data[key]['baseline']['Test Mean Absolute Percentage Error']
        except:
            return
        if b > data[key][index]['Test Mean Absolute Percentage Error']:
            string = str("<p style='color: #29C684;'> <b>This is better than baseline.</b></p>")
            st.markdown(string, unsafe_allow_html=True)
        else:
            string = str("<p style='color: #C63329;'> <b>This is worst than baseline.</b></p>")
            st.markdown(string, unsafe_allow_html=True)


    def f_add_xg_boost(self, data, key, index):
        string = str("<h4 style='text-align: center; color: #08568B;'><b>XG BOOST Analysis - " + index + "</b></h4>")
        st.markdown(string, unsafe_allow_html=True)
        st.image(data[key][index]['Image1'])
        st.image(data[key][index]['Image2'])
        st.image(data[key][index]['Image3'])
        st.write('Feature names:', data[key][index]['features_name'])
        st.write('Test Mean Absolute Error:', data[key][index]['Test Mean Absolute Error'])
        self.check_better_mae_baseline(data, key, index)
        st.write('Test Root Mean Squared Error:', data[key][index]['Test Root Mean Squared Error'])
        self.check_better_rmse_baseline(data, key, index)
        st.write('Test Mean Absolute Percentage Error:', data[key][index]['Test Mean Absolute Percentage Error'])
        self.check_better_mape_baseline(data, key, index)
        st.write('Accuracy:', data[key][index]['accuracy'])
        st.write(data[key][index]['Predictions'])
        st.line_chart(data[key][index]['Predictions'])
        #st.image(data[key][index]['diff_plot'])
        st.write("Correlation: y vs y-predicted")
        st.image('report/' + key + "/" + index + '_correlation.png')
        st.write('Time: ', data[key][index]['time(s)'], ' seconds')

    def f_algorithms(self):
        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader('Algorithms available')
        st.write('None yet')

    def f_metrics(self):
        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader('Metrics')

    def create_classification_section(self, data):
        st.markdown("<hr>", unsafe_allow_html=True)
        st.header('Classification')




