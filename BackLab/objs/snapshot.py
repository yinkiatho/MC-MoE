

class Snapshot:
    def __init__(self, inputs):
        # generally take a snapshot of the messages and other information
        self.snapshot_run = inputs.snapshot if (inputs.create_performance_file and inputs.snapshot) else False
        self.open_session_nav = {}
        self.close_session_nav = {}
        self.open_messages = {}
        self.close_messages = {}
        self.open_current_proportions = {}
        self.close_current_proportions = {}
        self.open_set_proportions = {}
        self.close_set_proportions = {}
        self.open_dollar_pnl = {}
        self.close_dollar_pnl = {}
        self.open_pct_pnl = {}
        self.close_pct_pnl = {}
        self.open_prices = {}
        self.close_prices = {}
        self.open_units = {}
        self.close_units = {}
        self.open_borrowing_amt = {}
        self.close_borrowing_amt = {}

    def ss_nav(self, bar_date, bar_type, nav):
        if (self.snapshot_run):
            if (bar_type == "Open"):
                self.open_session_nav[bar_date] = nav
            elif (bar_type == "Close"):
                self.close_session_nav[bar_date] = nav

        return 
    
    def ss_messages(self, bar_date, bar_type, stocks):
        if (self.snapshot_run):
            total_messages = []
            if (bar_type == "Open"):
                for ticker, o in stocks.items():
                    if o.message_on_open != "":
                        total_messages.append(f"[{ticker}] {o.message_on_open}")
                    
                final_message = "\n".join(map(str, total_messages))
                self.open_messages[bar_date] = final_message

            elif (bar_type == "Close"):
                for ticker, o in stocks.items():
                    if o.message_on_close != "":
                        total_messages.append(f"[{ticker}] {o.message_on_close}")
                    
                final_message = "\n".join(map(str, total_messages))
                self.close_messages[bar_date] = final_message

        return 
    
    def ss_proportions(self, bar_date, bar_type, stocks):
        if (self.snapshot_run):
            current_proportion_lst = []
            set_proportion_lst = []
            
            for ticker, o in stocks.items():
                ticker_mod = ticker.replace("=", "_")
                current_proportion_lst.append(f"{ticker_mod}={round(o.current_proportion+0.0,5)}")
                set_proportion_lst.append(f"{ticker_mod}={round(o.proportion+0.0,5)}")

            if (bar_type == "Open"):
                self.open_current_proportions[bar_date] = "/".join(current_proportion_lst)
                self.open_set_proportions[bar_date] = "/".join(set_proportion_lst)

            elif (bar_type == "Close"):
                self.close_current_proportions[bar_date] = "/".join(current_proportion_lst)
                self.close_set_proportions[bar_date] = "/".join(set_proportion_lst)
        return 

    def ss_pnl(self, bar_date, bar_type, stocks):
        if (self.snapshot_run):
            dollar_pnl_lst = []
            pct_pnl_lst = []

            for ticker, o in stocks.items():
                ticker_mod = ticker.replace("=", "_")
                dollar_pnl_lst.append(f"{ticker_mod}={round(o.pnl_dollar_hist[bar_date][bar_type]+0.0,4)}")
                pct_pnl_lst.append(f"{ticker_mod}={round(o.pnl_pct_hist[bar_date][bar_type]+0.0,7)}")

            if (bar_type == "Open"):
                self.open_dollar_pnl[bar_date] = "/".join(dollar_pnl_lst)
                self.open_pct_pnl[bar_date] = "/".join(pct_pnl_lst)

            elif (bar_type == "Close"):
                self.close_dollar_pnl[bar_date] = "/".join(dollar_pnl_lst)
                self.close_pct_pnl[bar_date] = "/".join(pct_pnl_lst)

        return 
    
    def ss_prices(self, bar_date, bar_type, stocks):
        if (self.snapshot_run):
            prices = []

            if (bar_type == "Open"):
                for ticker, o in stocks.items():
                    ticker_mod = ticker.replace("=", "_")
                    prices.append(f"{ticker_mod}={round(o.open[0],6)}")
                self.open_prices[bar_date] = "/".join(prices)
            elif (bar_type == "Close"):
                for ticker, o in stocks.items():
                    ticker_mod = ticker.replace("=", "_")
                    prices.append(f"{ticker_mod}={round(o.close[0],6)}")
                self.close_prices[bar_date] = "/".join(prices)

        return

    def ss_units(self, bar_date, bar_type, stocks):
        if (self.snapshot_run):
            total_units = []

            for ticker, o in stocks.items():
                ticker_mod = ticker.replace("=", "_")
                total_units.append(f"{ticker_mod}={o.units_holding}")

            if (bar_type == "Open"):
                self.open_units[bar_date] = "/".join(total_units)
            elif (bar_type == "Close"):
                self.close_units[bar_date] = "/".join(total_units)

        return
    
    def ss_borrowing_info(self, bar_date, bar_type, stocks):
        if (self.snapshot_run):
            total_borrowing_amt = []
            
            for ticker, o in stocks.items():
                ticker_mod = ticker.replace("=", "_")
                total_borrowing_amt.append(f"{ticker_mod}={o.borrowing_amt}")

            if (bar_type == "Open"):
                self.open_borrowing_amt[bar_date] = "/".join(total_borrowing_amt)
            elif (bar_type == "Close"):
                self.close_borrowing_amt[bar_date] = "/".join(total_borrowing_amt)

        return 
    


