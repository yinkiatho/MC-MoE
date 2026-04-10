import indicator_prerun
import numpy as np
from series import Series
from indicator_validity import indicator_validity

class Correlation:
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
        
        # indicator logic, remember that slicing of a series is same as slicing normal list
        if (self.active_bars < lookback):
            self.values.add(np.nan)
        else:
            start_idx = series1.length() - lookback
            end_idx = series1.length()
            corr_matrix = np.corrcoef([series1[start_idx:end_idx], series2[start_idx:end_idx]], ddof=1)
            corr_val = corr_matrix[0][1]
            self.values.add(corr_val)
        
        return 

    

