import preloads.load_all
import sys
import os 
from src.log import Log
from src.helper import timeit
from src.backtest_handler import BacktestHandler
from src.data_handler import DataHandler
from objs.inputs import Inputs
from datetime import datetime as dt
import time
import numpy as np
from indicator.sma import SMA

class Optimization:
    def __init__(self):
        self.run = False

        # example
        self.params = {'a': [0, 1, 2, 3],
                       'b': [6, 7, 8, 9]}

class Data:
    def __init__(self, log):
        self.tickers = ["^GSPC"]
        data_handler = DataHandler(self.tickers, start_date = "2010-04-01", end_date = "2025-01-01",
                                   interval = '1d', 
                                   reference_ticker = "^GSPC", method = "yfinance", log = log)

        # dont touch
        self.data_handler = data_handler
        self.stock_data = data_handler.stock_data
        self.reference_data = data_handler.reference_data
                
class BacktestLogic:
    def __init__(self, parameters = None):
        self.filename = os.path.basename(__file__)
        self.inputs = Inputs(initial_capital = 100000, leverage = 1, rebalance_proportion_diff = 0.10, commission = {"fix_per_trade": 0.0, "percent": 0.0006},
                    slippage = 0.0000, slippage_type = "percent", shorting_cost = 0.00, profit_taking = 0,
                    stop_loss = 0, min_reentry_bar = 0, price_filter = 0.001, rebalance_on_bar_open = True, rebalance_on_bar_close = False,
                    create_log = False, create_performance_file = True, snapshot = True, filename = self.filename)
        return

    def on_bar_open(self, stocks, bar, bar_type, date_time, date_time_series):
        # [open] is only updated at on_bar_open whereas [high/low/close] are not updated

        if (bar > 0):
            for ticker, o in stocks.items():
                o.proportion = 1

        return stocks
    
    def on_bar_close(self, stocks, bar, bar_type, date_time, date_time_series):
        # [high/low/open/close] are updated at on_bar_close


        return stocks

@timeit
def main():
    backtest_handler = BacktestHandler(BacktestLogic, Data, Optimization)
    backtest_handler.run()
    return 

if __name__ == "__main__":
    main()





    