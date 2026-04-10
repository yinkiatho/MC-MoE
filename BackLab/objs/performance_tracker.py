import numpy as np
import pandas as pd
import platform 
from datetime import datetime

class PerformanceTracker:
    def __init__(self, capital):
        # total inclusion of all tickers
        # inside all dict, there will be two attributes => Open, Close
        self.initial_capital = capital
        self.nav = capital
        self.prev_session_nav = capital
        self.free_cash = capital
        self.portfolio_dollar_pnl = {}
        self.portfolio_pct_pnl = {}
        self.bar_date_tracker = {}
        self.portfolio_close2close_pnl_dollar = {}
        self.portfolio_close2close_pnl_pct = {}
        self.cumulative_pnl_pct = 0
        self.daily_close_nav = {}
        self.total_net_exposure = 0
        self.start_date = None
        self.end_date = None
        self.num_of_trading_days = 0
        self.current_date = None

    def nav_update(self, date_time, bar, bar_type, stocks):
        
        holdings_nav = 0
        if (bar_type == "Open"):
            for ticker, o in stocks.items():
                if (not np.isnan(o.open[0])):
                    holdings_nav += o.units_holding * o.open[0]
        
        if (bar_type == "Close"):
            for ticker, o in stocks.items():
                if (not np.isnan(o.close[0])):
                    holdings_nav += o.units_holding * o.close[0]

        self.nav = holdings_nav + self.free_cash
        self.daily_close_nav[date_time] = self.nav
        self.portfolio_close2close_pnl_dollar[date_time] = (self.nav-self.daily_close_nav[self.bar_date_tracker[bar-1]]) if (bar > 0) else (self.nav-self.initial_capital)
        self.portfolio_close2close_pnl_pct[date_time] = (self.nav/self.daily_close_nav[self.bar_date_tracker[bar-1]] - 1) if (bar > 0) else (self.nav/self.initial_capital-1)
        self.cumulative_pnl_pct = (self.nav/self.initial_capital-1)
        return 

    def portfolio_update_on_price_change(self, bar, date_time, bar_type, stocks):
        ''''
        bar_type = Open / Close
        '''

        if (bar_type == "Open"):
            # only open price has been updated
            
            if (bar != 0):
                # we take previous close vs today's open 
                total_dollar_change_from_last_session = 0
                
                for ticker, o in stocks.items():
                    ticker_dollar_change_from_last_session = (o.open[0]-o.close[0]) * o.units_holding if (not np.isnan(o.close[0])) else 0
                    total_dollar_change_from_last_session += ticker_dollar_change_from_last_session
                    o.pnl_dollar_hist[date_time][bar_type] += ticker_dollar_change_from_last_session
                    o.pnl_pct_hist[date_time][bar_type] = o.pnl_dollar_hist[date_time][bar_type]/self.prev_session_nav
  
                self.portfolio_dollar_pnl[date_time][bar_type] += total_dollar_change_from_last_session
                self.portfolio_pct_pnl[date_time][bar_type] = self.portfolio_dollar_pnl[date_time][bar_type]/self.nav

            else:
                for ticker, o in stocks.items():
                    ticker_dollar_change_from_last_session = 0

                self.portfolio_dollar_pnl[date_time][bar_type] = 0
                self.portfolio_pct_pnl[date_time][bar_type] = 0

        if (bar_type == "Close"):
            # we take today's open vs today's close
            self.num_of_trading_days += 1 
            total_dollar_change_from_last_session = 0
            
            for ticker, o in stocks.items():
                ticker_dollar_change_from_last_session = (o.close[0]-o.open[0]) * o.units_holding if (not np.isnan(o.close[0]) and not np.isnan(o.open[0])) else 0
                total_dollar_change_from_last_session += ticker_dollar_change_from_last_session
                o.pnl_dollar_hist[date_time][bar_type] += ticker_dollar_change_from_last_session
                o.pnl_pct_hist[date_time][bar_type] = o.pnl_dollar_hist[date_time][bar_type]/self.prev_session_nav

            self.portfolio_dollar_pnl[date_time][bar_type] += total_dollar_change_from_last_session
            self.portfolio_pct_pnl[date_time][bar_type] = self.portfolio_dollar_pnl[date_time][bar_type]/self.nav

            self.portfolio_close2close_pnl_dollar[date_time] = (self.nav-self.daily_close_nav[self.bar_date_tracker[bar-1]]) if (bar > 0) else (self.nav-self.initial_capital)
            self.portfolio_close2close_pnl_pct[date_time] = (self.nav/self.daily_close_nav[self.bar_date_tracker[bar-1]] - 1) if (bar > 0) else (self.nav/self.initial_capital-1)
            self.daily_close_nav[date_time] = self.nav
            
        return stocks
    
    def portfolio_update_on_rebalance(self, bar, bar_type, date_time, stocks, commission_incurred, slippage_incurred, borrowing_amt_latest, ticker_exposure_change):

        for ticker, o in stocks.items():
            if (ticker in ticker_exposure_change.keys()):
                o.pnl_dollar_hist[date_time][bar_type] -= (commission_incurred[ticker] + slippage_incurred[ticker])
                o.borrowing_amt = borrowing_amt_latest[ticker]
                self.nav -= (commission_incurred[ticker] + slippage_incurred[ticker])
                self.free_cash -= (ticker_exposure_change[ticker] + commission_incurred[ticker] + slippage_incurred[ticker])

        for ticker, o in stocks.items():
            o.pnl_pct_hist[date_time][bar_type] = o.pnl_dollar_hist[date_time][bar_type]/self.prev_session_nav

        self.cumulative_pnl_pct = (self.nav/self.initial_capital-1)

        if (bar_type == "Close"):
            self.portfolio_close2close_pnl_dollar[date_time] = self.nav - self.daily_close_nav[self.bar_date_tracker[bar-1]] if bar > 0 else self.nav - self.initial_capital
            self.portfolio_close2close_pnl_pct[date_time] = round(self.nav/self.daily_close_nav[self.bar_date_tracker[bar-1]]-1, 15) if bar > 0 else self.nav/self.initial_capital - 1
            self.daily_close_nav[date_time] = self.nav

        return 
    
    def portfolio_session_initialization(self, date_time, bar, bar_type, stocks):
        if (bar == 0):
            self.start_date = date_time
        
        self.current_date = date_time
        self.end_date = self.current_date
        self.bar_date_tracker[bar] = self.current_date
        self.prev_session_nav = self.nav

        if (bar_type == "Open"):
            self.portfolio_dollar_pnl[date_time] = {}
            self.portfolio_pct_pnl[date_time] = {}

            for ticker, o in stocks.items():
                o.pnl_dollar_hist[date_time] = {}
                o.pnl_pct_hist[date_time] = {}

        # general 
        self.portfolio_dollar_pnl[date_time][bar_type] = 0
        self.portfolio_pct_pnl[date_time][bar_type] = 0

        for ticker, o in stocks.items():
            o.pnl_dollar_hist[date_time][bar_type] = 0
            o.pnl_pct_hist[date_time][bar_type] = 0

        return stocks

    def portfolio_update_on_shorting(self, bar, bar_type, date_time, stocks, daily_shorting_cost):
        # note that intraday shorting cost is not considered (this is happening in the real world too)
        # daily rate of shorting = shorting_cost*(1/252)

        if (bar_type == "Close"):
            return stocks  

        if (bar_type == "Open"):
            for ticker, o in stocks.items():
                # get short notional amount
                if (o.borrowing_amt > 0):
                    # remember that although this is the open session the o.close[0] is depending on previous' day close 
                    borrowing_cost = daily_shorting_cost * o.borrowing_amt
                    o.pnl_dollar_hist[date_time][bar_type] -= borrowing_cost
                    self.nav -= borrowing_cost
                    self.free_cash -= borrowing_cost
                    
            for ticker, o in stocks.items():        
                o.pnl_pct_hist[date_time][bar_type] = o.pnl_dollar_hist[date_time][bar_type]/self.nav
            
        return stocks
    
    def portfolio_update_on_dividend(self, bar_type, date_time, stocks, dividend_data):
        if (bar_type == "Close"):
            for ticker in dividend_data.keys():
                stocks[ticker].dividend_amt = 0.0
            return stocks
        
        if (bar_type == "Open"):
            no_action = True
            dt_obj = datetime(date_time.year, date_time.month, date_time.day)
            for ticker, dividend_class in dividend_data.items():
                if (len(dividend_class.agg_dividends) != 0 and dt_obj in dividend_class.agg_dividends):
                    # dividend is present today
                    no_action = False
                    total_dividend_amt = stocks[ticker].units_holding * dividend_class.agg_dividends[dt_obj]
                    stocks[ticker].pnl_dollar_hist[date_time][bar_type] += total_dividend_amt
                    stocks[ticker].dividend_amt = total_dividend_amt
                    self.nav += total_dividend_amt
                    self.free_cash += total_dividend_amt
                else:
                    stocks[ticker].dividend_amt = 0.0

            if (not no_action):
                for ticker, o in stocks.items():        
                    o.pnl_pct_hist[date_time][bar_type] = o.pnl_dollar_hist[date_time][bar_type]/self.nav
        return stocks
    

            
            



        
        
    