import indicator_prerun
import numpy as np
from series import Series
from indicator_validity import indicator_validity
from pct_change import PctChange
from variance import Variance
from covariance import Covariance

class Beta:
    def __init__(self):
        self.values = Series()
        self.active_bars = 0

        # indicator specific variables
        self.asset_pct_change = PctChange()
        self.mkt_pct_change = PctChange()
        self.mkt_var = Variance()
        self.asset_mkt_cov = Covariance()

    def __getitem__(self, index):
        return self.values[index]

    def __str__(self):
        return str(self.values)

    def __repr__(self):
        return repr(self.values)

    def length(self):
        return self.values.length()
        
    def update(self, asset_series, market_series, lookback):
        # check if the series values are non-nan before proceeding
        self.active_bars = indicator_validity(self.active_bars, asset_series, market_series)
        if self.active_bars == 0:
            self.values.add(np.nan)
            return

        # indicator logic
        self.asset_pct_change.update(asset_series, 1)
        self.mkt_pct_change.update(market_series, 1)
        self.mkt_var.update(self.asset_pct_change, lookback)
        self.asset_mkt_cov.update(self.asset_pct_change, self.mkt_pct_change, lookback)

        if (not np.isnan(self.mkt_var[0]) and not np.isnan(self.asset_mkt_cov[0])):
            self.values.add(self.asset_mkt_cov[0]/self.mkt_var[0])
        else:
            self.values.add(np.nan)
            
        return 
