import heapq
import numpy as np
from stock import initialize_stock
import copy

class StopLossOrders:
    def __init__(self, log):
        # tickers heap will keep the heap tree structure of lowest price of stop loss to highest, any selling of stocks will be carried out of lowest stop loss price first
        self.ticker_heap = {}
        self.ticker_heap_copy = {}
        self.log = log

    def rebalance_on(self, ticker, bar_type, price, current_units, units_to_rebal, stop_loss_pct):
        if (ticker not in self.ticker_heap):
            self.ticker_heap[ticker] = []

        if (stop_loss_pct > 0.0):
            # logic of insertion or removing
            if (current_units > 0) :
                if (units_to_rebal > 0):
                    stop_loss_price = price * (1-stop_loss_pct)
                    heapq.heappush(self.ticker_heap[ticker], (stop_loss_price, units_to_rebal))
                else:
                    if (current_units + units_to_rebal >= 0):
                        self.remove_top_heap_units(ticker, units_to_rebal)
                    else:
                        self.ticker_heap[ticker] = []
                        stop_loss_price = price * (1+stop_loss_pct)
                        heapq.heappush(self.ticker_heap[ticker], (stop_loss_price, current_units+units_to_rebal))
            else:
                if (units_to_rebal < 0):
                    stop_loss_price = price * (1+stop_loss_pct)
                    heapq.heappush(self.ticker_heap[ticker], (stop_loss_price, units_to_rebal))
                else:
                    if (current_units + units_to_rebal <= 0):
                        self.remove_top_heap_units(ticker, units_to_rebal)
                    else:
                        self.ticker_heap[ticker] = []
                        stop_loss_price = price * (1-stop_loss_pct)
                        heapq.heappush(self.ticker_heap[ticker], (stop_loss_price, current_units+units_to_rebal))
                
    def remove_top_heap_units(self, ticker, units_to_rebal):
        
        units_removed = 0
        
        if (units_to_rebal == 0):
            return 
        elif (units_to_rebal > 0):
            
            while (units_removed < units_to_rebal):
                top_heap = self.ticker_heap[ticker][0]
                top_heap_units = top_heap[1] # negative values
                top_heap_price = top_heap[0]
                heapq.heappop(self.ticker_heap[ticker])
                units_removed += (-top_heap_units)

            if (units_removed > units_to_rebal):
                placement_units = units_to_rebal - units_removed
                stop_loss_price = top_heap_price
                heapq.heappush(self.ticker_heap[ticker], (stop_loss_price, placement_units))
        else:
            
            while (units_removed > units_to_rebal):
                top_heap = self.ticker_heap[ticker][0]
                top_heap_units = top_heap[1] # positive values
                top_heap_price = top_heap[0]
                heapq.heappop(self.ticker_heap[ticker])
                units_removed += (-top_heap_units)

            if (units_removed < units_to_rebal):
                placement_units = units_to_rebal - units_removed
                stop_loss_price = top_heap_price
                heapq.heappush(self.ticker_heap[ticker], (stop_loss_price, placement_units))
        return 

    def stop_loss_check(self, stocks, bar, bar_type, nav, min_reentry_bars):
        self.ticker_heap_copy = copy.deepcopy(self.ticker_heap)
        for ticker, o in stocks.items():
            if (bar_type == "Open"):
                price_to_compare = o.open[0]
            elif (bar_type == "Close"):
                price_to_compare = o.close[0]

            if (np.isnan(price_to_compare) or not (ticker in self.ticker_heap_copy) or len(self.ticker_heap_copy[ticker]) == 0):
                continue
            
            units_changed_due_to_stoploss = 0
            if (o.units_holding > 0):
                highest_heap = self.ticker_heap_copy[ticker][0]
                while (price_to_compare <= highest_heap[0]):
                    units_changed_due_to_stoploss -= highest_heap[1]
                    self.log.stop_loss_trigger("StopLossTrigger", ticker, bar_type, price_to_compare, highest_heap[0], highest_heap[1])
                    heapq.heappop(self.ticker_heap_copy[ticker])
                    if (len(self.ticker_heap_copy[ticker]) != 0):
                        highest_heap = self.ticker_heap_copy[ticker][0]
                    else:
                        break

                if (units_changed_due_to_stoploss == -o.units_holding):
                    proportion_change_due_to_stoploss = -o.rebalance_proportion
                else:
                    proportion_change_due_to_stoploss = (units_changed_due_to_stoploss * price_to_compare)/nav

                if (o.rebalance_proportion > o.current_proportion):
                    pending_rebalance_proportion = o.rebalance_proportion + proportion_change_due_to_stoploss
                else:
                    pending_rebalance_proportion = min(o.current_proportion+proportion_change_due_to_stoploss, o.rebalance_proportion)

                if (units_changed_due_to_stoploss != 0 and o.rebalance_proportion > o.current_proportion + proportion_change_due_to_stoploss):
                    min_reentry_bars.flag(ticker, pending_rebalance_proportion, bar, units_changed_due_to_stoploss)

                o.rebalance_proportion = pending_rebalance_proportion
                
            else:
                highest_heap = self.ticker_heap_copy[ticker][0]

                while (price_to_compare >= highest_heap[0]):
                    units_changed_due_to_stoploss -= highest_heap[1]
                    self.log.stop_loss_trigger("StopLossTrigger", ticker, bar_type, price_to_compare, highest_heap[0], highest_heap[1])
                    heapq.heappop(self.ticker_heap_copy[ticker])
                    if (len(self.ticker_heap_copy[ticker]) != 0):
                        highest_heap = self.ticker_heap_copy[ticker][0]
                    else:
                        break

                if (units_changed_due_to_stoploss == -o.units_holding):
                    proportion_change_due_to_stoploss = -o.rebalance_proportion
                else:
                    proportion_change_due_to_stoploss = (units_changed_due_to_stoploss * price_to_compare)/nav

                if (o.rebalance_proportion < o.current_proportion):
                    pending_rebalance_proportion = o.rebalance_proportion + proportion_change_due_to_stoploss
                else:
                    pending_rebalance_proportion = max(o.current_proportion+proportion_change_due_to_stoploss, o.rebalance_proportion)

                if (units_changed_due_to_stoploss != 0 and o.rebalance_proportion < o.current_proportion + proportion_change_due_to_stoploss):
                    min_reentry_bars.flag(ticker, pending_rebalance_proportion, bar, units_changed_due_to_stoploss)

                o.rebalance_proportion = pending_rebalance_proportion

        return stocks, min_reentry_bars
            

if __name__ == "__main__":
    #testing
    SLO = StopLossOrders()

    stocks = initialize_stock(["SPY", "QQQ"])
    print(stocks)
    stocks["SPY"].close.add(100)
    stocks["QQQ"].close.add(200)
    stocks["SPY"].proportion = -0.2
    stocks["QQQ"].proportion = -0.8
    stocks["SPY"].rebalance_proportion = -0.2
    stocks["QQQ"].rebalance_proportion = -0.8
    stocks["SPY"].units_holding = -10
    stocks["QQQ"].units_holding = -20

    """
    cur_units_spy = 0
    rebal_units = 10
    SLO.rebalance_on("SPY", 100, cur_units_spy, rebal_units, 0.05)
    cur_units_spy += rebal_units
    print(SLO.ticker_heap)

    cur_units_qqq = 0
    rebal_units = 20
    SLO.rebalance_on("QQQ", 200, cur_units_qqq, rebal_units, 0.05)
    cur_units_qqq += rebal_units
    print(SLO.ticker_heap)

    # spy close decreases to 94
    # qqq close at 195
    stocks["SPY"].close.add(94)
    stocks["QQQ"].close.add(195)

    stocks["SPY"].current_proportion = 0.1942
    stocks["QQQ"].current_proportion = 0.8058

    stocks = SLO.stop_loss_check(stocks, "Close", 4840)
    print(f"spy proportion: {stocks['SPY'].rebalance_proportion}")
    print(f"qqq proportion: {stocks['QQQ'].rebalance_proportion}")
    """

    cur_units_spy = 0
    rebal_units = -10
    SLO.rebalance_on("SPY", 100, cur_units_spy, rebal_units, 0.05)
    cur_units_spy += rebal_units
    print(SLO.ticker_heap)

    cur_units_qqq = 0
    rebal_units = -20
    SLO.rebalance_on("QQQ", 200, cur_units_qqq, rebal_units, 0.05)
    cur_units_qqq += rebal_units
    print(SLO.ticker_heap)

    # spy close decreases to 101
    # qqq close at 211
    stocks["SPY"].close.add(101)
    stocks["QQQ"].close.add(209)

    stocks["SPY"].current_proportion = 0.1942
    stocks["QQQ"].current_proportion = 0.8058

    stocks = SLO.stop_loss_check(stocks, "Close", 4770)
    print(f"spy proportion: {stocks['SPY'].rebalance_proportion}")
    print(f"qqq proportion: {stocks['QQQ'].rebalance_proportion}")

    stocks["SPY"].close.add(101)
    stocks["QQQ"].close.add(210)

    cur_units_qqq = -20
    rebal_units = -5
    SLO.rebalance_on("QQQ", 210, cur_units_qqq, rebal_units, 0.05)
    cur_units_qqq += rebal_units
    print(SLO.ticker_heap)

    stocks["SPY"].proportion = -0.2
    stocks["QQQ"].proportion = -1.0
    stocks["SPY"].rebalance_proportion = -0.2
    stocks["QQQ"].rebalance_proportion = -1.0
    stocks["SPY"].units_holding = -10
    stocks["QQQ"].units_holding = -25

    stocks = SLO.stop_loss_check(stocks, "Close", 4790)
    print(f"spy proportion: {stocks['SPY'].rebalance_proportion}")
    print(f"qqq proportion: {stocks['QQQ'].rebalance_proportion}")



    


            



