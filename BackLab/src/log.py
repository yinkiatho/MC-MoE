import logging
import os 
import platform

class Log:
    def __init__(self, create_log = True, filename = "unknown", optimization_run = False):
        self.create_log = create_log
        self.optimization_run = optimization_run
        if (self.create_log):
            if (platform.system() == "Linux"):
                # linux
                self.logger = setup_logger(f"logs/{filename.replace('.py', '')}.log")
            else:
                #windows
                self.logger = setup_logger(f"logs\\{filename.replace('.py', '')}.log")

    def to_screen(self, msg):
        if (self.create_log):
            # print(msg)
            self.logger.info(msg)

    def print_session(self, bar, bar_type, rebalance_this_sesion):
        if (rebalance_this_sesion):
            self.to_screen(f"Session: {bar_type}, Bar: {bar}")

    def download_progress(self, class_name, type, method):
        if (type == "start"):
            self.to_screen(f"[{class_name}] Downloading data from {method}")
        if (type == "end"):
            self.to_screen(f"[{class_name}] Completed data download.\n")

    def progress_to_screen(self, bar, total_bar, every = 0.1):
        if (not self.optimization_run):
            print_bar_mod = int(round(total_bar*every,0)+1)
            if (bar == 0 or bar % print_bar_mod == 0 or bar == total_bar - 1):
                bar_percent = int(round(100 * ((bar+1)/total_bar)))
                max_bar = int(100/5)
                cur_bar = int(bar_percent/5)
                empty_space = max_bar - cur_bar - 1
                empty_space = empty_space+1 if (cur_bar == 0) else empty_space

                if (cur_bar != max_bar):
                    print(f"Backtest Progress: {bar_percent}%  {cur_bar * '|'}{empty_space * ' '}{'|'}")
                else:
                    cur_bar -= 1
                    print(f"Backtest Progress: {bar_percent}% {cur_bar * '|'}{empty_space * ' '}{'|'}")
        return 
    
    def alert_missing_data_point(self, class_name, ticker, date_time, bar_component):
        self.to_screen(f"[{class_name}] Alert! Missing data point spotted on ticker: {ticker}, date_time: {date_time}, bar_component: {bar_component}. Previous bar price is used.")

    def print_stock_message(self, stocks, bar_type):
        if (bar_type == "Open"):
            for ticker, o in stocks.items():
                #print(f"o is: {o}", type(o))
                if (o.message_on_open != ""):
                    self.to_screen(f"[MessageOnOpen / {o.name}] " + o.message_on_open)

        if (bar_type == "Close"):
            for ticker, o in stocks.items():
                if (o.message_on_close != ""):
                    self.to_screen(f"[MessageOnClose / {o.name}] " + o.message_on_close)

        return

    def print_stock_attributes(self, date_time, stocks):
        for ticker, o in stocks.items():
            self.to_screen(f"[StockAttributes / {ticker}] Open: {round(o.open[0],4)}, Close: {round(o.close[0],4)}, Proportion: {round(o.proportion,3)}, RebalanceProportion: {round(o.rebalance_proportion,3)}, CurrentProportion: {round(o.current_proportion,3)}, UnitsHolding: {o.units_holding}, PnLOpen(%) : {round(o.pnl_pct_hist[date_time]['Open']*100,4)}%, PnLClose(%) : {round(o.pnl_pct_hist[date_time]['Close']*100,4)}%")

    def sod_divider(self, date_time):
        # start of day divider
        self.to_screen(f"[DATE TIME: {date_time}] -----------------------------------------------------------------------------------------------------------------------------")
    
    def eod_divider(self, date_time):
        # end of day divider
        self.to_screen("\n")

    def rebalance_process(self, class_name, ticker, bar_type, current_proportion, rebalance_proportion,  cur_qty, target_qty, units_to_rebal, commission, slippage):
        self.to_screen(f"[{class_name}-{bar_type}] Ticker: {ticker}, CurrentProportion: {'{:.3f}'.format(current_proportion)}, Rebalance Proportion: {r'{:.3f}'.format(rebalance_proportion)}, Current Qty: {cur_qty}, Target Qty: {target_qty}, Units to Rebalance: {units_to_rebal}, Commission Incurred: {'{:.3f}'.format(commission)}, Slippage Incurred: {'{:.3f}'.format(slippage)}")

    def eod_print_out(self, performance_tracker, date_time):
        self.to_screen("~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        self.to_screen(f"End-of-day NAV: {'{:.2f}'.format(performance_tracker.nav)}")
        self.to_screen(f"One-day PnL: {'{:.3f}'.format(performance_tracker.portfolio_close2close_pnl_pct[date_time]*100)}%")
        self.to_screen(f"Cumulative PnL: {'{:.3f}'.format(performance_tracker.cumulative_pnl_pct*100)}%")
        self.to_screen("~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    def create_stop_loss_order(self, class_indicator, ticker, bar_type, price, heap):
        if (len(heap) > 0):
            self.to_screen(f"[{class_indicator}] Ticker: {ticker}, bar_type: {bar_type}, current price: {price}, StopLossHeap: {heap}")

    def stop_loss_trigger(self, class_indicator, ticker, bar_type, price, stop_loss_price, expected_quantity_reduced):
        self.to_screen(f"[{class_indicator}] Stop loss triggered on ticker: {ticker}, BarType: {bar_type}, CurrentPrice: {price}, StopLossPrice: {r'{:.6f}'.format(stop_loss_price)}, ExpectedQuantityToReduce: {expected_quantity_reduced}")

    def create_min_reentry_bar_flag(self, class_indicator, ticker, bar, reentry_bars, proportion_limit, units_changed_due_to_stoploss):
        if (units_changed_due_to_stoploss < 0):
            self.to_screen(f"[{class_indicator}] Triggered Min Reentry Bar flag on {ticker}, CurrentBars: {bar}, ReentryBars: {reentry_bars}, ProportionLimit: {r'{:.3f}'.format(proportion_limit)}; limit placed on increasing exposure")
        else:
            self.to_screen(f"[{class_indicator}] Triggered Min Reentry Bar flag on {ticker}, CurrentBars: {bar}, ReentryBars: {reentry_bars}, ProportionLimit: {r'{:.3f}'.format(proportion_limit)}; limit placed on decreasing exposure")

    def min_reentry_bar_flag_check(self, class_indicator, ticker, bar, reentry_bars, rebalance_proportion, proportion_limit, units_changed_due_to_stoploss):
        if (units_changed_due_to_stoploss < 0):
            self.to_screen(f"[{class_indicator}] Min Reentry Bar Flag Check on {ticker}, CurrentBars: {bar}, ReentryBars: {reentry_bars}, RebalanceProportion: {r'{:.3f}'.format(rebalance_proportion)}, unable to increase exposure more than: {r'{:.3f}'.format(proportion_limit)}")
        else:
            self.to_screen(f"[{class_indicator}] Min Reentry Bar Flag Check on {ticker}, CurrentBars: {bar}, ReentryBars: {reentry_bars}, RebalanceProportion: {r'{:.3f}'.format(rebalance_proportion)}, unable to decrease exposure less than: {r'{:.3f}'.format(proportion_limit)}")

    def price_filtering_check(self, class_indicator, ticker, bar_type, current_price, filtering_price, rebalance_proportion, filtered_proportion):
        self.to_screen(f"[{class_indicator}] Ticker: {ticker}, BarType: {bar_type}, Price: {current_price}, FilteringPriceBelow: {filtering_price}, RebalanceProportion: {r'{:.3f}'.format(rebalance_proportion)},, FilteredProportion: {filtered_proportion}")



def setup_logger(log_file):
    
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # Create a formatter
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Clear the log file if it exists
    if os.path.exists(log_file):
        open(log_file, 'w').close()

    return logger