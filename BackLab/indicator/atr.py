import numpy as np
from series import Series
from indicator.sma import SMA
from indicator_validity import indicator_validity

class ATR:
    def __init__(self):
        self.values = Series()
        self.active_bars = 0

        # indicator specific variables
        self.tr = Series()
        self.average_tr = SMA()

    def __getitem__(self, index):
        return self.values[index]

    def __str__(self):
        return str(self.values)
    
    def __repr__(self):
        return repr(self.values)

    def length(self):
        return self.values.length()
    
    def update(self, high_series, low_series, close_series, lookback):
        # check if the series values are non-nan before proceeding
        self.active_bars = indicator_validity(self.active_bars, high_series, low_series, close_series)
        
        if self.active_bars <= 1:
            self.values.add(np.nan)
            return 

        # indicator logic
        tr_val = np.maximum(
            np.maximum(high_series[0] - low_series[0], np.abs(high_series[0] - close_series[1])),
            np.abs(low_series[0] - close_series[1])
        )
        self.tr.add(tr_val)
        self.average_tr.update(self.tr, lookback)
        self.values.add(self.average_tr[0])

        return 
    
       