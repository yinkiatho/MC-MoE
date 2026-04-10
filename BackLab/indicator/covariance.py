import indicator_prerun
import numpy as np
from series import Series
from indicator_validity import indicator_validity

class Covariance:
    def __init__(self):
        self.values = Series()
        self.active_bars = 0

        # indicator specific variables

    def __getitem__(self, index):
        return self.values[index]
    
    def __str__(self):
        return str(self.values)
    
    def __repr__(self):
        return repr(self.values)

    def length(self):
        return self.values.length()
    
    def update(self, series1, series2, lookback):
        # check if the series values is non nan before proceeding
        self.active_bars = indicator_validity(self.active_bars, series1, series2)
        if (self.active_bars == 0):
            self.values.add(np.nan)
            return 
        ##############################################################

        # indicator logic
        """
        series1/series2 = raw values like stock price/indicators
        indicator: take in series values, spit out the rolling variance of the series
        """
        if (self.active_bars < lookback):
            self.values.add(np.nan)
        else:
            start_idx1 = series1.length() - lookback
            end_idx1 = series1.length()

            start_idx2 = series2.length() - lookback
            end_idx2 = series2.length()

            computing_series = np.stack((series1[start_idx1:end_idx1], series2[start_idx2:end_idx2]), axis = 0)
            cov_matrix = np.cov(computing_series)
            self.values.add(cov_matrix[0][1])

        return 
        