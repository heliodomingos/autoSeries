import pandas as pd


class TimeSeries(pd.DataFrame):

    def read_csv(self, filename):
        return TimeSeries(pd.read_csv(filename))
        #self._data = pd.read_csv(filename)

    def read_txt(self, filename, delimiter=';'):
        return TimeSeries(pd.read_csv(filename, delimiter))

    def read_xls(self, filename):
        self._data = pd.read_excel(filename)

    def read_sql(self, filename):
        self._data = pd.read_sql(filename)

