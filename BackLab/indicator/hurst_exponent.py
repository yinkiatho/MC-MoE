#import indicator_prerun
import numpy as np
from series import Series
from indicator_validity import indicator_validity
from hurst import compute_Hc
import random
import matplotlib.pyplot as plt

class HurstExponent:
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

    def update(self, series_values, lookback):
        # check if the series values is non nan before proceeding
        if (lookback < 100):
            raise Exception("Series values length must be greater or equal to 100 due to the use of hurst package.")

        self.active_bars = indicator_validity(self.active_bars, series_values)
        if (self.active_bars == 0):
            self.values.add(np.nan)
            return 
        ##############################################################

        # indicator logic
        if (self.active_bars < lookback):
            self.values.add(np.nan)
            return
        else:
            start_idx = series_values.length() - lookback
            end_idx = series_values.length()
           
            hurst_exponent, _, _ = compute_Hc(series_values[start_idx:end_idx], kind='price', simplified=True)
            self.values.add(hurst_exponent)
        return 
        
if __name__ == "__main__":
    random.seed(10)
    hurst = HurstExponent()
    s = Series()

    z = random.random()
    s.add(z)
    hurst.update(s, 100)
    print(hurst)
   
    for i in range(1000):
        z = 1+(random.random()-0.5)*0.01
        s.add(s[0]*z)
        k = max(100, s.length())
        hurst.update(s, k)

    print(s)
    print("-----------------------------------")
    print(hurst)

    plt.plot(s.values)
    plt.show()
        




