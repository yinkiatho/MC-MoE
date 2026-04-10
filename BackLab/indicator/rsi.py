import numpy as np
from series import Series
from indicator_validity import indicator_validity
from indicator.conditional_average_change import ConditionalAverageChange
from indicator.pct_change import PctChange

class RSI:
    def __init__(self):
        self.values = Series()
        self.active_bars = 0

        # indicator specific variables
        self.pct_change = PctChange()
        self.conditional_average_change_positive = ConditionalAverageChange()
        self.conditional_average_change_negative = ConditionalAverageChange()
        
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
        
        if self.active_bars < lookback:
            self.values.add(np.nan)
            return 

        # indicator logic
        self.pct_change.update(series_values)
        self.conditional_average_change_positive.update(self.pct_change, lookback, "larger_than", divide = "by_lookback")
        self.conditional_average_change_negative.update(self.pct_change, lookback, "smaller_than", divide = "by_lookback")
        
        if (np.isnan(self.conditional_average_change_positive[0]) or np.isnan(self.conditional_average_change_negative[0])):
            self.values.add(np.nan)
            return 
    
        if (self.conditional_average_change_negative[0] == 0):
            self.values.add(100)
        else:
            rsi_val = 100 - 100/(1+self.conditional_average_change_positive[0]/abs(self.conditional_average_change_negative[0]))
            self.values.add(rsi_val)

        return 
    