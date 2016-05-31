import numpy as np
import pandas as pd
from datetime import datetime

date_format_str = '%Y-%m-%d %H:%M:%S'

from datetime import datetime

def process_date_column(data_sframe):
    """Split the 'datetime' column of a given sframe"""
    parsed_date = pd.DatetimeIndex(data_sframe['datetime'])
    data_sframe['year'] = parsed_date.year
    data_sframe['month'] = parsed_date.month
    data_sframe['day'] = parsed_date.day
    data_sframe['hour'] = parsed_date.hour
    # minute information is not very useful for the prediction is it?
    #    data_sframe['minute'] = parsed_date.minute
    data_sframe['weekday'] = parsed_date.weekday

    del data_sframe['datetime']



# We'll impute missing values using the median for numeric columns and the most
# common value for string columns.
# This is based on some nice code by 'sveitser' at http://stackoverflow.com/a/25562948
from sklearn.base import TransformerMixin


class DataFrameImputer(TransformerMixin):
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].median() for c in X],
            index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.fill)