import numpy as np
from series import Series
from indicator_validity import indicator_validity
from indicator.ema import EMA

# Implementation taken from https://www.investopedia.com/terms/t/tsi.asp

class TrueStrengthIndex:
    def __init__(self):
        self.values = Series()
        self.active_bars = 0

        # indicator specific variables
        # Double Smoothed PC
        self.PC = Series()
        self.PC_FS = EMA()
        self.PC_SS = EMA()

        # Double Smoothed Absolute PC
        self.APC = Series()
        self.APC_FS = EMA()
        self.APC_SS = EMA()

    def __getitem__(self, index):
        return self.values[index]

    def __str__(self):
        return str(self.values)

    def __repr__(self):
        return repr(self.values)

    def length(self):
        return self.values.length()
    
    def update(self, close_prices, first_ts_lookback = 25, second_ts_lookback = 13):
        # check if the series values are non-nan before proceeding
        self.active_bars = indicator_validity(self.active_bars, close_prices)
        
        if self.active_bars <= 1:
            self.values.add(np.nan)
            return 

        self.PC.add(close_prices[0]-close_prices[1])
        self.PC_FS.update(self.PC, 1, first_ts_lookback)
        self.PC_SS.update(self.PC_FS, 1, second_ts_lookback)

        self.APC.add(abs(close_prices[0]-close_prices[1]))
        self.APC_FS.update(self.APC, 1, first_ts_lookback)
        self.APC_SS.update(self.APC, 1, second_ts_lookback)

        if (not np.isnan(self.PC[0]) and not np.isnan(self.PC_FS[0]) and not np.isnan(self.PC_SS[0]) and not np.isnan(self.APC[0]) and
            not np.isnan(self.APC_FS[0]) and not np.isnan(self.APC_SS[0])):
            self.values.add(100*(self.PC_SS[0]/self.APC_SS[0]))
        else:
            self.values.add(np.nan)

        return 
    
