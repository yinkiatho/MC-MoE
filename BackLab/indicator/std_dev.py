import indicator_prerun
import numpy as np
from series import Series
from indicator_validity import indicator_validity

class StdDev:
    def __init__(self):
        self.values = Series()
        self.active_bars = 0

    def __getitem__(self, index):
        return self.values[index]
    
    def __str__(self):
        return str(self.values)
    
    def __repr__(self):
        return repr(self.values)

    def length(self):
        return self.values.length()

    def update(self, series_values, lookback):
        # check if the series values is non nan before proceeding
        self.active_bars = indicator_validity(self.active_bars, series_values)
        if (self.active_bars == 0):
            self.values.add(np.nan)
            return 
        ##############################################################

        # indicator logic
        """
        series values = raw values like stock price
        indicator: take in series values, spit out the one-bar volatility (this can be editted using sqrt(252) or sqrt(52))
        """
        
        if (self.active_bars >= lookback):
            start_idx = series_values.length() - lookback
            end_idx = series_values.length()

            std_dev_val = np.std(series_values[start_idx:end_idx], ddof = 1)
            self.values.add(std_dev_val)
        else:
            self.values.add(np.nan)

        return 
        