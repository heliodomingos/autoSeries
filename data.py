from load_data_module import load_data
from data_profiling_module import explore_data
from TimeSeries import *


def forecast_data():
    # household_power_consumption = load_data(TimeSeries(), filename_list[4], target_variables[4])
    #stock_price = load_data(TimeSeries(), filename_list[2], target_variables[2])
    #temperatures = load_data(TimeSeries(), filename_list[0], target_variables[0])
    #sales = load_data(TimeSeries(), "data/sales.csv", "close")
    spy = load_data(TimeSeries(), "data/SPY.csv", "close")
    google = load_data(TimeSeries(), "data/GOOG.csv", "close")
    tesla = load_data(TimeSeries(), "data/TSLA.csv", "close")


    nio = load_data(TimeSeries(), "data/NIO.csv", "close")
    san = load_data(TimeSeries(), "data/SAN.csv", "close")
    orcl = load_data(TimeSeries(), "data/ORCL.csv", "close")

    # furniture = load_data(TimeSeries(), "data/superstore.xls", "Sales")
    return {
        # 'Household power consumption': {
        #    'raw_data': household_power_consumption,
        #    'insights': explore_data(household_power_consumption, 'Household power consumption')
        # },
        #'Stock Price': {
        #    'raw_data': stock_price,
        #    'insights': explore_data(stock_price, 'Stock Price')
        #},
        # 'Temperatures': {
        #    'raw_data': temperatures,
        #    'insights': explore_data(temperatures, 'Temperatures')
        #},
        #'SALES': {
        #    'raw_data': sales,
        #    'insights': explore_data(sales, 'SALES')
        #},
        'SP500 index': {
            'raw_data': spy,
            'insights': explore_data(spy, 'SP500 index')
        },
        'Example Dataset': {
            'raw_data': google,
            'insights': explore_data(google, 'Example Dataset')
        },
        'Tesla Stock': {
            'raw_data': tesla,
            'insights': explore_data(tesla, 'Tesla Stock')
        },



        'Nio Stock': {
            'raw_data': nio,
            'insights': explore_data(nio, 'Nio Stock')
        },
        'Santander Stock': {
            'raw_data': san,
            'insights': explore_data(san, 'Santander Stock')
        },
        'Oracle Stock': {
            'raw_data': orcl,
            'insights': explore_data(orcl, 'Oracle Stock')
        }
        #,
        #'Furniture Sales': {
         #   'raw_data': furniture,
          #  'insights': explore_data(furniture, 'Furniture Sales')
        #}
    }
