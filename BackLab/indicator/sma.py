import indicator_prerun
import numpy as np
from series import Series
from indicator_validity import indicator_validity

class SMA:
    def __init__(self):
        self.values = Series()
        self.active_bars = 0

        # indicator specific variables
        self.total_sum = 0

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
        self.total_sum += series_values[0]
        
        if (self.active_bars > lookback):
            self.total_sum -= series_values[lookback]

        self.values.add(self.total_sum/min(lookback, self.active_bars))
        return 
        

if __name__ == "__main__":
    sma = SMA()
    s = Series([np.nan])
    sma.update(s, 5)

    print(sma)

    s.add(10)
    sma.update(s, 5)

    print(sma)
    
    s.add(11)
    sma.update(s, 5)

    print(sma)
    
    s.add(12)
    sma.update(s, 5)

    print(sma)
    
    s.add(13)
    sma.update(s, 5)

    print(sma)

    s.add(14)
    sma.update(s, 5)

    print(sma)

    s.add(15)
    sma.update(s, 5)

    print(sma)

    s.add(16)
    sma.update(s, 5)

    print(sma)
