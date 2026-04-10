import numpy as np
from series import Series
from indicator.sma import SMA
from indicator_validity import indicator_validity

class MaxSeries:
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
    
    def update(self, series_values, lookback):
        # check if the series values are non-nan before proceeding
        self.active_bars = indicator_validity(self.active_bars, series_values)
        
        if self.active_bars == 0:
            self.values.add(np.nan)
            return 

        # indicator logic
        end_idx = series_values.values.length()
        start_idx = series_values.values.length()-lookback
        self.values.add(max(series_values[max(start_idx,0):end_idx]))

        return 
    
       