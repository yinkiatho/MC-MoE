import indicator_prerun
import numpy as np
from series import Series
from indicator_validity import indicator_validity
from pct_change import PctChange

class Volatility:
    def __init__(self):
        self.values = Series()
        self.active_bars = 0

        # indicator specific variables
        self.pct_change = PctChange()

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
        self.pct_change.update(series_values, 1)

        if (self.active_bars >= lookback+1):
            start_idx = self.pct_change.length() - lookback
            end_idx = self.pct_change.length()

            vol = np.std(self.pct_change[start_idx:end_idx], ddof = 1)
            self.values.add(vol)
        else:
            self.values.add(np.nan)

        return 
        