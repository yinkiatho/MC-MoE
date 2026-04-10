import indicator_prerun
import numpy as np
from series import Series
from indicator_validity import indicator_validity

class EMA:
    def __init__(self):
        self.values = Series()
        self.active_bars = 0

        # indicator specific variables
        self.prev_ema = 0

    def __getitem__(self, index):
        return self.values[index]
    
    def __str__(self):
        return str(self.values)
    
    def __repr__(self):
        return repr(self.values)

    def length(self):
        return self.values.length()
    
    def update(self, series_values, smoothing, lookback):
        # check if the series values are non-nan before proceeding
        self.active_bars = indicator_validity(self.active_bars, series_values)
        if self.active_bars == 0:
            self.values.add(np.nan)
            return 

        # indicator logic
        value = series_values[0]

        if self.active_bars == 1:
            ema = value
        else:
            ema = (smoothing/(1+lookback))* value + (1 - (smoothing/(1+lookback))) * self.prev_ema

        self.values.add(ema)
        self.prev_ema = ema

        return 