try:
    from objs.series import Series
except:
    from series import Series
import numpy as np
from collections import deque

class Stock:
    def __init__(self, name, index):
        self.name = name
        self.index = index
        self.lowest_price = 999999999999999999999
        self.highest_price = 0
        self.units_holding = 0
        self.is_active = False
        self.proportion = 0 # user set
        self.rebalance_proportion = 0 # proportion after going through filters
        self.current_proportion = 0 # proportion updated on open and close bar
        self.restricted_trading_bars = 0
        self.date =  Series() 
        self.open = Series()  
        self.close = Series()
        self.high = Series()
        self.low = Series()
        self.volume = Series()
        self.profit_taking_price = None
        self.stop_loss_orders = [] # the sum of units of this order type must be equivalent to units holding
        self.stop_loss_price = None
        self.trades_done = 0
        self.borrowing_amt = 0
        self.dividend_amt = 0
        self.pnl_dollar_hist = {}
        self.pnl_pct_hist = {}
        self.message_on_open = ""
        self.message_on_close = ""

def initialize_stock(tickers):
    trading_list = {}
    counter = 0

    for ticker in tickers:
        trading_list[ticker] = Stock(ticker, counter)
        counter += 1

    return trading_list

def readjust_current_proportion(stocks, nav, bar_type):
    for ticker, o in stocks.items():
        if (bar_type == "Open"):
            if (not np.isnan(o.open[0])):
                o.current_proportion = (o.open[0]*o.units_holding)/nav
        elif (bar_type == "Close"):
            if (not np.isnan(o.close[0])):
                o.current_proportion = (o.close[0]*o.units_holding)/nav

    return stocks

def set_proportion_to_rebalance_weight(stocks, leverage):
    for ticker, o in stocks.items():
        o.rebalance_proportion = o.proportion * leverage

    return stocks

def reset_pending_msg(stocks, bar_type):
    if (bar_type == "Open"):
        for ticker, o in stocks.items():
            o.message_on_open = ""

    if (bar_type == "Open"):
        for ticker, o in stocks.items():
            o.message_on_close = ""

    return stocks
    

    


    
