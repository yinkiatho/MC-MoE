import numpy as np
from series import Series
from indicator_validity import indicator_validity

class IBS:
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
    
    def update(self, high_series, low_series, close_series):
        # check if the series values are non-nan before proceeding
        self.active_bars = indicator_validity(self.active_bars, high_series, low_series, close_series)
        
        if self.active_bars == 0:
            self.values.add(np.nan)
            return 

        # indicator logic
        if (high_series[0]==low_series[0]):
            ibs_val = 1
        else:
            ibs_val = (close_series[0]-low_series[0])/(high_series[0]-low_series[0])

        self.values.add(ibs_val)

        return 
    
       