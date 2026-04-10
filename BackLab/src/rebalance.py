import numpy as np
import pandas as pd
import math
from src.log import Log
from src.trade import Order

class Rebalance:
    def __init__(self, inputs, log):
        self.inputs = inputs 
        self.rebalance_error = RebalanceError()
        self.log = log

    def run(self, date_time, bar_type, stocks, performance_tracker, stop_loss_orders, total_orders_book):
        # bar_type = Open/Close
        commission_incurred = {}
        slippage_incurred = {}
        borrowing_amt_latest = {}
        ticker_exposure_change = {}
        for ticker, o in stocks.items():
            self.rebalance_error.check(stocks, bar_type)

            # rebalance criterias
            rebalance_criteria1 = (o.rebalance_proportion != 0.0 and o.current_proportion == 0.0) or (o.rebalance_proportion == 0.0 and o.current_proportion != 0.0) 

            # delta rebalancing
            rebalance_criteria2 = (o.current_proportion != 0.0 and abs(o.rebalance_proportion-o.current_proportion) > self.inputs.rebalance_proportion_diff)
            
            if (rebalance_criteria1 or rebalance_criteria2):
                # carry out rebalance to the most accurate shares
                current_qty = o.units_holding
                if (o.rebalance_proportion > 0):
                    if (bar_type == "Open"):
                        target_qty = math.floor((performance_tracker.nav*o.rebalance_proportion)/o.open[0])
                    elif (bar_type == "Close"):
                        target_qty = math.floor((performance_tracker.nav*o.rebalance_proportion)/o.close[0])
                elif (o.rebalance_proportion < 0):
                    if (bar_type == "Open"):
                        target_qty = math.ceil((performance_tracker.nav*o.rebalance_proportion)/o.open[0])
                    elif (bar_type == "Close"):
                        target_qty = math.ceil((performance_tracker.nav*o.rebalance_proportion)/o.close[0])
                else:
                    target_qty = 0

                units_to_rebal = target_qty - current_qty

            else:
                units_to_rebal = 0

            if (units_to_rebal != 0):
                # we need to do buy/sell shares to achieve target proportion as close as possible
                if (bar_type == "Open"):
                    placement_volume = 0 if (o.volume.length == 0) else o.volume[0]
                    commission = Commission.get(self.inputs.commission, abs(units_to_rebal), o.open[0], placement_volume*21)
                    slippage = Slippage.get(self.inputs.slippage, self.inputs.slippage_type, abs(units_to_rebal), o.open[0], placement_volume*21)
                    borrowing_amt = BorrowingAmtUpdate.get(units_to_rebal, o.units_holding, o.borrowing_amt, o.open[0])
                    exposure_change = units_to_rebal*o.open[0]
                    stop_loss_orders.rebalance_on(ticker, bar_type, o.open[0], o.units_holding, units_to_rebal, self.inputs.stop_loss)
                    current_price = o.open[0]
                    total_orders_book[ticker].add_order(Order(ticker=ticker, date_init=date_time, quantity=units_to_rebal, price=o.open[0]))

                elif (bar_type == "Close"):
                    commission = Commission.get(self.inputs.commission, abs(units_to_rebal), o.close[0], o.volume[0]*21)
                    slippage = Slippage.get(self.inputs.slippage, self.inputs.slippage_type, abs(units_to_rebal), o.close[0], o.volume[0]*21)
                    borrowing_amt = BorrowingAmtUpdate.get(units_to_rebal, o.units_holding, o.borrowing_amt, o.close[0])
                    exposure_change = units_to_rebal*o.close[0]
                    stop_loss_orders.rebalance_on(ticker, bar_type, o.close[0], o.units_holding, units_to_rebal, self.inputs.stop_loss)
                    current_price = o.close[0]
                    total_orders_book[ticker].add_order(Order(ticker=ticker, date_init=date_time, quantity=units_to_rebal, price=o.close[0]))

                self.log.rebalance_process(self.__class__.__name__, ticker, bar_type, o.current_proportion, o.rebalance_proportion,  current_qty, target_qty, units_to_rebal, commission, slippage)
                self.log.create_stop_loss_order("StopLossOrderHeap", ticker, bar_type, current_price, stop_loss_orders.ticker_heap[ticker])
                o.units_holding += units_to_rebal
                commission_incurred[ticker] = commission 
                slippage_incurred[ticker] = slippage
                borrowing_amt_latest[ticker] = borrowing_amt
                ticker_exposure_change[ticker] = exposure_change

        return stocks, commission_incurred, slippage_incurred, borrowing_amt_latest, ticker_exposure_change, stop_loss_orders, total_orders_book

class Commission:
    @staticmethod
    def get(commission, share_qty, share_price, volume):
        # implement commission structure
        return commission["percent"]*(share_qty*share_price) + commission["fix_per_trade"]

class Slippage:
    @staticmethod
    def get(slippage, slippage_type, share_qty, share_price, volume):
        # implement slippage structure
        if (slippage_type == "percent"):
            return slippage * (share_qty*share_price)
        else:
            return 0
        
class RebalanceError:
    def __init__(self):
        return 
    
    def check(self, stocks, bar_type):
        for ticker, o in stocks.items():
            if ((bar_type == "Open" and np.isnan(o.open[0]) or (bar_type == "Close" and np.isnan(o.close[0]))) and o.rebalance_proportion != 0.0):
                raise Exception(f"[{self.__class__.__name__}] {bar_type} Price of {ticker} is NaN, but rebalance proportion is non-zero.")
       
        return 
    
class BorrowingAmtUpdate:
    @staticmethod
    def get(units_to_rebal, units_holding_before_rebal, borrowing_amt, current_price):
        if (units_to_rebal + units_holding_before_rebal >= 0):
            borrowing_amt = 0.0
        else:
            if (units_holding_before_rebal >= 0):
                borrowing_amt = abs((units_holding_before_rebal+units_to_rebal)*current_price)
            else:
                # current state is short
                if (units_to_rebal > 0):
                    # reduce short
                    borrowing_amt = borrowing_amt * (1+units_to_rebal/units_holding_before_rebal)
                else:
                    # increasing short
                    borrowing_amt = borrowing_amt + abs(units_to_rebal + current_price)

        return borrowing_amt