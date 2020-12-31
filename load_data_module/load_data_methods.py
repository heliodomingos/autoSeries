import pandas as pd
import time


def find_date_col(time_series):
    date_string = ["DATE", "date", "Date"]
    time_col = ["Time", "time"]
    res = []
    start = time.time()

    for e in time_series.keys():
        if e in date_string:
            res = [e] + res
        elif e in time_col:
            res += [e]
    if res == []:
        print("Datetime column not found. Please rename to 'date'.")

    if len(res) > 1:
        time_series['date'] = pd.to_datetime(time_series[res[0]] + ' ' + time_series[res[1]])
        time_series.drop(res, axis=1, inplace=True)
        time_series = time_series.set_index('date')
    else:
        time_series = time_series.set_index(res[0])
        time_series.index = pd.to_datetime(time_series.index, exact=True)
    end = time.time()
    print('time: ', end - start)
    return time_series


def load_data_from_xls(time_series, filename):
    df = pd.read_excel(filename)
    furniture = df.loc[df['Category'] == 'Furniture']
    furniture.drop('Category', axis=1, inplace=True)
    #furniture['Order Date'].min(), furniture['Order Date'].max()
    #cols = ['Row ID', 'Order ID', 'Ship Date', 'Ship Mode', 'Customer ID', 'Customer Name', 'Segment', 'Country',
    #        'City', 'State', 'Postal Code', 'Region', 'Product ID', 'Category', 'Sub-Category', 'Product Name',
    #        'Quantity', 'Discount', 'Profit']
    date_type_col = []
    for e in furniture.columns:
        if 'Date' in e or 'date' in e or 'dates' in e or 'Dates' in e:
            date_type_col.append(e)

    furniture.drop(['Row ID', 'Order ID', 'Customer ID', 'Customer Name', 'Segment', 'City', 'State', 'Postal Code', 'Region', 'Product ID', 'Sub-Category'], axis=1, inplace=True)
    furniture = furniture.sort_values('Order Date')
    # furniture = furniture.groupby('Order Date')['Sales'].sum().reset_index()
    furniture = furniture.set_index('Order Date')
    i = list(furniture.columns).index('Sales')
    col = furniture.columns.tolist()
    if i > 0:
        c = furniture['Sales'].copy()
        furniture.drop('Sales', axis=1, inplace=True)
        furniture.insert(0, column='Sales', value=c)
    return furniture


def load_data_from_tsv(time_series, filename):
    print(filename)
    time_series = pd.read_table(filename)
    return time_series


def load_data_from_txt(time_series, filename, target_variables):
    """
    :param time_series: a time series structure
    :param filename: string that indicates the path to the csv file
    :return: pandas dataframe composed by a datetime index and a target value
    """
    time_series = time_series.read_txt(filename, delimiter=';')
    time_series = find_date_col(time_series)
    drop_cols = list(set(time_series.columns) - set([target_variables]))
    time_series.drop(drop_cols, axis=1, inplace=True)
    print('Extra columns droped')
    for col in time_series.columns:
        time_series[col] = pd.to_numeric(time_series[col], errors='coerce')
    time_series = time_series.dropna()
    print('load from txt done')
    return time_series.sort_index()


def load_data_from_csv(time_series, filename, target_variables):
    """
    :param time_series: a time series structure
    :param filename: string that indicates the path to the csv file
    :return: pandas dataframe composed by a datetime index and a target value
    """

    time_series = time_series.read_csv(filename)
    time_series.columns = time_series.columns.str.lower()
    time_series = find_date_col(time_series)

    #df.index = pd.to_datetime(df.index, exact=True)
    # time_series.index = pd.to_datetime(time_series.index, exact=True)
    drop_cols = list(set(time_series.columns) - set([target_variables]))
    time_series.drop(drop_cols, axis=1, inplace=True)
    return time_series.sort_index()


def load_data_from_sql(database):
    """
    :param database: string with the datawarehouse name
    :return: pandas dataframe composed by a datetime index and a target value
    """

    #from sqlalchemy import create_engine
    #sqlEngine = create_engine('mysql+pymysql://root:@127.0.0.1', pool_recycle=3600)
    #dbConnection = sqlEngine.connect()
    #df = pd.read_sql("select * from fact_table", dbConnection)
    #dbConnection.close()
    #return df

# more complex
# df = pd.read_csv('stock.csv', index_col='Date', usecols=['Date','Real Price'])
# df.head()
