import indicator_prerun
import numpy as np
from series import Series
from indicator_validity import indicator_validity

class Ratio:
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

    def update(self, series1, series2):
        # check if the series values is non nan before proceeding
        self.active_bars = indicator_validity(self.active_bars, series1, series2)
        if (self.active_bars == 0):
            self.values.add(np.nan)
            return 
        ##############################################################

        # indicator logic
        self.values.add(series1[0]/series2[0])
        return 
        
