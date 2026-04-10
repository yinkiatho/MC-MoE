import numpy as np
from series import Series
from indicator_validity import indicator_validity
from indicator.atr import ATR

class Supertrend:
    def __init__(self):
        self.values = Series()
        self.active_bars = 0

        # indicator specific variables
        self.atr = ATR()
        self.historical_high = Series()
        self.historical_low = Series()

    def __getitem__(self, index):
        return self.values[index]

    def __str__(self):
        return str(self.values)

    def __repr__(self):
        return repr(self.values)

    def length(self):
        return self.values.length()
    
    def update(self, high_series, low_series, close_series, lookback = 20, multiplier = 3):
        # check if the series values are non-nan before proceeding
        self.active_bars = indicator_validity(self.active_bars, high_series, low_series, close_series)
        
        if self.active_bars == 0:
            self.values.add(np.nan)
            return 

        # indicator logic
        self.atr.update(high_series, low_series, close_series, lookback)
        series_start = high_series.length()-lookback
        series_end = high_series.length()
        self.historical_high.add(max(high_series[series_start:series_end]))
        self.historical_low.add(min(low_series[series_start:series_end]))
        
        if (not np.isnan(self.atr[0])):
            self.values.add((self.historical_high[0]+self.historical_low[0])*0.5 + multiplier*self.atr[0])
        else:
            self.values.add(np.nan)
    
        return 
    
