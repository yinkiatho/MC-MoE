import numpy as np

class PriceFilters:
    def __init__(self, log, filter_price_below = 0.2):
        self.filter_price_below = filter_price_below
        self.log = log

    def check(self, stocks, bar_type):
        if (bar_type == "Open"):
            for ticker, o in stocks.items():
                if (not np.isnan(o.open[0])):
                    if (o.open[0] < self.filter_price_below and o.rebalance_proportion != 0):
                        self.log.price_filtering_check("PriceFilterExecution", ticker, bar_type, o.open[0], self.filter_price_below, o.rebalance_proportion, 0)
                        o.rebalance_proportion = 0
                        
        if (bar_type == "Close"):
            for ticker, o in stocks.items():
                if (not np.isnan(o.close[0])):
                    if (o.close[0] < self.filter_price_below and o.rebalance_proportion != 0):
                        self.log.price_filtering_check("PriceFilterExecution", ticker, bar_type, o.close[0], self.filter_price_below, o.rebalance_proportion, 0)
                        o.rebalance_proportion = 0
        
        return stocks