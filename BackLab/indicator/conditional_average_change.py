import numpy as np
from series import Series
from indicator_validity import indicator_validity

class ConditionalAverageChange:
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
    
    def update(self, series_values, lookback, method = "larger_than", divide = "by_count"):
        # method = [larger_than, smaller_than], divide = [by_count, by_lookback]
        # check if the series values are non-nan before proceeding
        if not (method == "larger_than" or method == "smaller_than"):
            raise Exception("ConditionalAverageChange's method is not larger_than or smaller_than")

        self.active_bars = indicator_validity(self.active_bars, series_values)
        
        if (self.active_bars < lookback):
            self.values.add(np.nan)
            return 

        # indicator logic
        sum_values = 0
        sum_count = 0
        
        if (method == "larger_than"):
            for i in range(lookback):
                if (series_values[i] > 0):
                    sum_values += series_values[i]
                    sum_count += 1
        else:
            for i in range(lookback):
                if (series_values[i] < 0):
                    sum_values += series_values[i]
                    sum_count += 1

        if (sum_count != 0):
            if (divide == "by_count"):
                self.values.add(sum_values/sum_count)
            else:
                self.values.add(sum_values/lookback)
        else:
            self.values.add(0)
        
        return 
    