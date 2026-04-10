class Inputs:
    def __init__(self, initial_capital = 100000, leverage = 1, rebalance_proportion_diff = 0, commission = {"fix_per_trade": 0.38, "percent": 0.000436}, 
                 slippage = 0.0, slippage_type = "percent", shorting_cost = 0.0, profit_taking = 0, stop_loss = 0, min_reentry_bar = 0,
                 price_filter = 0.1, rebalance_on_bar_open = True, rebalance_on_bar_close = True, create_log = True, create_performance_file = True,
                 snapshot = False, filename = "unknown"):
        
        self.initial_capital = initial_capital
        
        # nav = stocks_holding + free cash
        self.leverage = leverage
        self.rebalance_proportion_diff = rebalance_proportion_diff
        self.commission = commission
        self.slippage = slippage
        self.slippage_type = slippage_type
        self.shorting_cost = shorting_cost
        self.daily_shorting_cost = shorting_cost*(1/252) 
        self.profit_taking = profit_taking  
        self.stop_loss = stop_loss
        self.min_reentry_bar = min_reentry_bar
        self.price_filter = price_filter
        self.rebalance_on_bar_open = rebalance_on_bar_open
        self.rebalance_on_bar_close = rebalance_on_bar_close
        self.create_log = create_log
        self.create_performance_file = create_performance_file
        self.snapshot = snapshot
        self.filename = filename.replace(".py", "")
        self.csv_folder = None

