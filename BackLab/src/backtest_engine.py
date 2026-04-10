from objs.series import Series
from objs.performance_tracker import PerformanceTracker
from objs.stock import readjust_current_proportion
from objs.stock import set_proportion_to_rebalance_weight
from objs.stock import reset_pending_msg
from src.rebalance import Rebalance
from objs.stoploss_orders import StopLossOrders
from objs.min_reentry_bar import MinReentryBars
from objs.price_filters import PriceFilters
from objs.snapshot import Snapshot
from objs.file_export import FileExport
from src.trade import TotalOrdersBook
import logging

class BacktestEngine:
    def __init__(self, backtest_logic, data, inputs, log):
        self.backtest_logic = backtest_logic
        self.inputs = inputs
        self.data_handler = data.data_handler
        self.stock_data = data.stock_data
        self.stocks = self.data_handler.stocks
        self.log = log
        self.performance_tracker = PerformanceTracker(inputs.initial_capital)
        self.min_reentry_bars = MinReentryBars(self.inputs.min_reentry_bar, self.log)
        self.stop_loss_orders = StopLossOrders(self.log)
        self.price_filtering = PriceFilters(self.log, filter_price_below = inputs.price_filter)
        self.rebalance = Rebalance(self.inputs, self.log)
        self.snapshot = Snapshot(self.inputs)
        self.total_orders_book = TotalOrdersBook(tickers = list(self.stocks.keys()))

    def run(self):
        # start by getting the looping index
        date_time_series = Series()
        reference_ticker = self.data_handler.reference_ticker
        bars_to_loop = list(self.data_handler.reference_data[reference_ticker].index)
        total_bars = len(bars_to_loop)
        bar = 0
        
        for bar_date_time in bars_to_loop:
            self.log.sod_divider(bar_date_time)
            date_time_series.add(bar_date_time)
            is_new_trading_day = True if (bar >= 1 and date_time_series[1].day != bar_date_time.day) else False
            
            # execute price update and portfolio update on bar open
            bar_type = "Open"
            self.log.print_session(bar, bar_type, self.inputs.rebalance_on_bar_open)
            self.stocks = self.performance_tracker.portfolio_session_initialization(bar_date_time, bar, bar_type, self.stocks)
            self.stocks = self.performance_tracker.portfolio_update_on_shorting(bar, bar_type, bar_date_time, self.stocks, self.inputs.daily_shorting_cost) if (is_new_trading_day) else self.stocks
            self.stocks = self.data_handler.update_price(self.stocks, bar_date_time, ["Open"], self.stock_data)
            self.stocks = self.performance_tracker.portfolio_update_on_price_change(bar, bar_date_time, bar_type, self.stocks)
            self.performance_tracker.nav_update(bar_date_time, bar, bar_type, self.stocks)
            self.stocks = readjust_current_proportion(self.stocks, self.performance_tracker.nav, bar_type)

            if (self.inputs.rebalance_on_bar_open):
                # execute logic on bar open
                self.stocks = self.backtest_logic.on_bar_open(self.stocks, bar, bar_type, bar_date_time, date_time_series)
                self.log.print_stock_message(self.stocks, bar_type)
                self.stocks = set_proportion_to_rebalance_weight(self.stocks, self.inputs.leverage)
                # execute order type action on bar open
                self.stocks = self.price_filtering.check(self.stocks, bar_type)
                self.stocks, self.min_reentry_bars = self.stop_loss_orders.stop_loss_check(self.stocks, bar, bar_type, self.performance_tracker.nav, self.min_reentry_bars)
                self.stocks = self.min_reentry_bars.check_flag(bar, self.stocks)
                # self.stocks = OrderTypeAction.profit_taking_filter(self.stocks)
                
                # execute rebalance on bar open
                self.stocks, commission_incurred, slippage_incurred, borrowing_amt_latest, ticker_exposure_change, self.stop_loss_orders, self.total_orders_book = self.rebalance.run(bar_date_time, bar_type, self.stocks, self.performance_tracker, self.stop_loss_orders, self.total_orders_book)

                # update performance on bar open after rebalance
                self.performance_tracker.portfolio_update_on_rebalance(bar, bar_type, bar_date_time, self.stocks, commission_incurred, slippage_incurred, borrowing_amt_latest, ticker_exposure_change)
                self.stocks = readjust_current_proportion(self.stocks, self.performance_tracker.nav, bar_type)

            # take a snapshot of the event happening in open session
            self.snapshot.ss_prices(bar_date_time, bar_type, self.stocks)
            self.snapshot.ss_units(bar_date_time, bar_type, self.stocks)
            self.snapshot.ss_messages(bar_date_time, bar_type, self.stocks)
            self.snapshot.ss_nav(bar_date_time, bar_type, self.performance_tracker.nav)
            self.snapshot.ss_borrowing_info(bar_date_time, bar_type, self.stocks)
            self.snapshot.ss_proportions(bar_date_time, bar_type, self.stocks)
            self.snapshot.ss_pnl(bar_date_time, bar_type, self.stocks)

            reset_pending_msg(self.stocks, bar_type)
            # end of bar open session

            # execute price update and portfolio update on bar close (new session)
            bar_type = "Close"
            self.log.print_session(bar, bar_type, self.inputs.rebalance_on_bar_close)
            self.stocks = self.performance_tracker.portfolio_session_initialization(bar_date_time, bar, bar_type, self.stocks)
            self.stocks = self.performance_tracker.portfolio_update_on_shorting(bar, bar_type, bar_date_time, self.stocks, self.inputs.daily_shorting_cost) if (is_new_trading_day) else self.stocks
            self.stocks = self.data_handler.update_price(self.stocks, bar_date_time, ["High", "Low", "Close", "Volume"], self.stock_data)
            self.stocks = self.performance_tracker.portfolio_update_on_price_change(bar, bar_date_time, bar_type, self.stocks)
            self.performance_tracker.nav_update(bar_date_time, bar, bar_type, self.stocks)
            self.stocks = readjust_current_proportion(self.stocks, self.performance_tracker.nav, bar_type)

            if (self.inputs.rebalance_on_bar_close):
                # execute logic on bar open
                self.stocks = self.backtest_logic.on_bar_close(self.stocks, bar, bar_type, bar_date_time, date_time_series)
                self.log.print_stock_message(self.stocks, bar_type)
                
                self.stocks = set_proportion_to_rebalance_weight(self.stocks, self.inputs.leverage)

                # execute order type action on bar close
                self.stocks = self.price_filtering.check(self.stocks, bar_type)
                self.stocks, self.min_reentry_bars = self.stop_loss_orders.stop_loss_check(self.stocks, bar, bar_type, self.performance_tracker.nav, self.min_reentry_bars)
                self.stocks = self.min_reentry_bars.check_flag(bar, self.stocks)
                # self.stocks = OrderTypeAction.profit_taking_filter(self.stocks)
                
                # execute rebalance on bar close
                self.stocks, commission_incurred, slippage_incurred, borrowing_amt_latest, ticker_exposure_change, self.stop_loss_orders, self.total_orders_book = self.rebalance.run(bar_date_time, bar_type, self.stocks, self.performance_tracker, self.stop_loss_orders, self.total_orders_book)
                
                # update performance on bar close after rebalance 
                self.performance_tracker.portfolio_update_on_rebalance(bar, bar_type, bar_date_time, self.stocks, commission_incurred, slippage_incurred, borrowing_amt_latest, ticker_exposure_change)
                self.stocks = readjust_current_proportion(self.stocks, self.performance_tracker.nav, bar_type)

            # take a snapshot of the event happening in close session
            self.snapshot.ss_prices(bar_date_time, bar_type, self.stocks)
            self.snapshot.ss_units(bar_date_time, bar_type, self.stocks)
            self.snapshot.ss_messages(bar_date_time, bar_type, self.stocks)
            self.snapshot.ss_nav(bar_date_time, bar_type, self.performance_tracker.nav)
            self.snapshot.ss_borrowing_info(bar_date_time, bar_type, self.stocks)
            self.snapshot.ss_proportions(bar_date_time, bar_type, self.stocks)
            self.snapshot.ss_pnl(bar_date_time, bar_type, self.stocks)
            
            reset_pending_msg(self.stocks, bar_type)
            # end of bar close session 

            self.log.print_stock_attributes(bar_date_time, self.stocks)
            self.log.eod_print_out(self.performance_tracker, bar_date_time)
            self.log.progress_to_screen(bar, total_bars)
            self.log.eod_divider(bar_date_time)
            bar += 1

        # print(self.snapshot.open_messages)
        # self.performance_tracker.export_to_csv(self.inputs.filename, folder=self.inputs.csv_folder)
        FileExport.run(self.inputs, self.performance_tracker, self.snapshot)
        # self.total_orders_book.print_trades()
        
        return

class OrderTypeAction:
    @staticmethod
    def stop_loss_filter(stocks):
        
        return stocks
    
    @staticmethod
    def profit_taking_filter(stocks):

        return stocks
    
    @staticmethod
    def price_threshold_filter(stocks):

        return stocks
    
    





    


        



    

    
