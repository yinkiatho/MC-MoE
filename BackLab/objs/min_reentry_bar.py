class MinReentryBars:
    def __init__(self, min_reentry_bars, log):
        """
        if you are flagged once by min reentry bars, you are barred from increasing exposure of the current direction.
        e.g. you hold 60% DBS, and stop loss class has ordered you to take off 20% weightage (left over 40%), you are not allowed to increase positive exposure for N bars, decreasing weights exposure is still possible
        e.g. you hold -60% DBS, and stop loss class has ordered you to take off -20% weightage (left over -40%), you are not allowed to increase negative exposure for N bars, increasing weights exposure is still possible
        """
        self.reentry_bars_pending = {}
        self.min_reentry_bars = min_reentry_bars
        self.log = log
        return 

    def flag(self, ticker, rebalance_proportion, bar, units_changed_due_to_stoploss):
        # flag item includes user can only trades >= bar_entered_flag + min reentry_bars
        if (self.min_reentry_bars > 0):
            flag_item = {"bars_entered": bar, "reentry_bars": bar + self.min_reentry_bars + 1,
                        "flagged_weight_limit": rebalance_proportion, "units_changed_due_to_stoploss": units_changed_due_to_stoploss}
            
            self.reentry_bars_pending[ticker] = flag_item
            self.log.create_min_reentry_bar_flag("MinReentryBarsFlag", ticker, bar, bar + self.min_reentry_bars + 1, rebalance_proportion, units_changed_due_to_stoploss)
        return 
    
    def check_flag(self, bar, stocks):
        if (self.min_reentry_bars > 0):
            reentry_bars_pending_copy = self.reentry_bars_pending.copy()
            for ticker, reentry_details in reentry_bars_pending_copy.items():
                reentry_bars = reentry_details["reentry_bars"]
                if (bar >= reentry_bars):
                    del self.reentry_bars_pending[ticker]
                else:
                    if (reentry_details["units_changed_due_to_stoploss"] < 0):
                        if (stocks[ticker].rebalance_proportion > reentry_details["flagged_weight_limit"]):
                            self.log.min_reentry_bar_flag_check("MinReentryBarsFlagCheck", ticker, bar, reentry_bars, stocks[ticker].rebalance_proportion, reentry_details["flagged_weight_limit"], reentry_details["units_changed_due_to_stoploss"])
                            stocks[ticker].rebalance_proportion = reentry_details["flagged_weight_limit"]
                    else:
                        if (stocks[ticker].rebalance_proportion < reentry_details["flagged_weight_limit"]):
                            self.log.min_reentry_bar_flag_check("MinReentryBarsFlagCheck", ticker, bar, reentry_bars, stocks[ticker].rebalance_proportion, reentry_details["flagged_weight_limit"], reentry_details["units_changed_due_to_stoploss"])
                            stocks[ticker].rebalance_proportion = reentry_details["flagged_weight_limit"]

        return stocks
                
            
    


   
        
    


