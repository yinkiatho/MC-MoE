import pandas as pd
import numpy as np
from scipy.stats import norm

class TradeStatistics:
    def __init__(self, return_df):
        self.df = return_df
    
    def run(self):
            columns_needed = ["OpenSessionNav", "OpenUnitsHolding", "OpenDollarPnl", "CloseSessionNav", "CloseUnitsHolding", "CloseDollarPnl"]
            all_columns_present = True

            for col in columns_needed:
                if (col not in self.df.columns):
                    all_columns_present = False
                    break
            
            # only calculate if we have the stats
            if (all_columns_present):
                # we want to return a dataframe that contains the ticker, number of trade made, total profit loss made (%),
                # portion of the trade in portfolio returns, duration holding, win rate of a trade, p-value of significance of trade

                # initial capital = first trading day of open session - OpenDollarPnl for all tickers 
                initial_capital = self.df.iat[0, self.df.columns.get_loc("OpenSessionNav")]
                initial_dollar_change = self.string_to_dict(self.df.iat[0, self.df.columns.get_loc("OpenDollarPnl")])
                
                # initial capital adjustment
                for ticker, dollar_change in initial_dollar_change.items():
                    initial_capital -= dollar_change
                
                active_state, unit_state, proportion_state, trades, trades_initial_nav, trade_num, trade_dir = self.initialize_state_and_trade()
                sessions = [["OpenSessionNav", "OpenUnitsHolding", "OpenSetProportions", "OpenDollarPnl"], ["CloseSessionNav", "CloseUnitsHolding", "CloseSetProportions", "CloseDollarPnl"]]
                for i in range(self.df.shape[0]):
                    # loop for two sessions:
                    for session_items in sessions:
                        session_nav = self.df.iat[i, self.df.columns.get_loc(session_items[0])]
                        current_units = self.string_to_dict(self.df.iat[i, self.df.columns.get_loc(session_items[1])])
                        current_proportions = self.string_to_dict(self.df.iat[i, self.df.columns.get_loc(session_items[2])])
                        session_dollar_pnl = self.string_to_dict(self.df.iat[i, self.df.columns.get_loc(session_items[3])])

                        active_state, trade_num, trades, trades_initial_nav, unit_state, proportion_state, trade_dir = self.on_new_session(session_nav, current_units,
                                                                                                                                current_proportions, session_dollar_pnl,
                                                                                                                                active_state, unit_state,
                                                                                                                                proportion_state, trades,
                                                                                                                                trades_initial_nav, trade_num,
                                                                                                                                trade_dir) 

                ts_df = self.create_ts_df(trade_num, trades, trades_initial_nav, trade_dir)
                return ts_df
            else:
                return None

    def initialize_state_and_trade(self):
        ticker_info = self.string_to_dict(self.df.iat[0, self.df.columns.get_loc("OpenDollarPnl")])
        active_state = {}
        unit_state = {}
        proportion_state = {}
        trades = {}
        trades_initial_nav = {}
        trade_dir = {}
        trade_num = {}
        for ticker in ticker_info.keys():
            active_state[ticker] = False
            unit_state[ticker] = 0
            proportion_state[ticker] = 0
            trades[ticker] = {} # record the dollar amt
            trades_initial_nav[ticker] = {} # for each ticker record the nav of start trading
            trade_dir[ticker] = {}
            trade_num[ticker] = 0
        return active_state, unit_state, proportion_state, trades, trades_initial_nav, trade_num, trade_dir

    def string_to_dict(self, string_return):
        indicative_returns = string_return.split('/')
        ticker_returns = {}
        for pair in indicative_returns:
            key, value = pair.split('=')
            ticker_returns[key] = float(value)  # Convert value to float if needed
        return ticker_returns
    
    def on_new_session(self, session_nav, current_units, current_proportions, session_dollar_pnl, active_state, unit_state,
                       proportion_state, trades, trades_initial_nav, trade_num, trade_dir):
        for ticker, units in current_units.items():
            if (active_state[ticker]):
                # if active state at beginning of session then record the info
                trades[ticker][trade_num[ticker]].append(session_dollar_pnl[ticker])

            # state change
            condition1 = unit_state[ticker] != units and units != 0 and current_proportions[ticker] != proportion_state[ticker]
            condition2 = (unit_state[ticker] > 0 and units < 0) or (unit_state[ticker] < 0  and units > 0)

            if (condition1 or condition2):
                trade_num[ticker] += 1
                trades[ticker][trade_num[ticker]] = []
                trades_initial_nav[ticker][trade_num[ticker]] = session_nav

                unit_state[ticker] = units
                proportion_state[ticker] = current_proportions[ticker]
                active_state[ticker] = True

                if (units > 0):
                    trade_dir[ticker][trade_num[ticker]] = 1
                else:
                    trade_dir[ticker][trade_num[ticker]] = -1

            elif (unit_state[ticker] != units and units == 0):
                active_state[ticker] = False
                unit_state[ticker] = units
                proportion_state[ticker] = current_proportions[ticker]

            elif (unit_state[ticker] != units and ((unit_state[ticker] > 0 and units > 0) or (unit_state[ticker] < 0 and units < 0))):
                unit_state[ticker] = units
                proportion_state[ticker] = current_proportions[ticker]

        return active_state, trade_num, trades, trades_initial_nav, unit_state, proportion_state, trade_dir

    def create_ts_df(self, trade_num, trades, trades_initial_nav, trade_dir):
        # col details: ticker
        # row details: num of trade, average pnl (%) per trade, average win rate given long, average return given long,
        # average win rate given short, average return given short, holding period (per trade, morning session and end session count as 0.5 days)

        ts_df = pd.DataFrame(index=["NumOfTotalTrades", "NumOfLongTrades", "NumOfShortTrades", 
                                    "WinRateOfTrades(%)", "WinRateGivenLong(%)", "WinRateGivenShort(%)",
                                    "AveragePnlPctPerTrade(%)", "AveragePnlPctGivenLong(%)", "AveragePnlPctGivenShort(%)",
                                    "AveragePnlPctGivenWin(%)", "AveragePnlPctGivenLose(%)", 
                                    "AverageHoldingPeriodPerTrade(Bar)", "AverageHoldingPeriodGivenLong(Bar)", "AverageHoldingPeriodGivenShort(Bar)",
                                    "p-value"],
                             columns=["Total"]+list(trade_num.keys()))
        total_long_pnl_pct = []
        total_long_periods = []
        total_short_pnl_pct = []
        total_short_periods = []

        for ticker in trade_num.keys():
            ts_df.loc["NumOfTotalTrades", ticker] = len(trades[ticker])
            
            # for each trade we evaluate 
            long_pnl_pct = []
            long_periods = []
            short_pnl_pct = []
            short_periods = []
            for trade_no, dollar_pnl_lst in trades[ticker].items():
                holding_period = len(dollar_pnl_lst)*0.5 # we are recording per session
                dollar_pnl = sum(dollar_pnl_lst)
                
                starting_trade_nav = trades_initial_nav[ticker][trade_no]
                pct_pnl = dollar_pnl/starting_trade_nav

                if (trade_dir[ticker][trade_no] == 1):
                    long_pnl_pct.append(pct_pnl)
                    long_periods.append(holding_period)
                else:
                    short_pnl_pct.append(pct_pnl)
                    short_periods.append(holding_period)

            total_long_pnl_pct += long_pnl_pct
            total_long_periods += long_periods
            total_short_pnl_pct += short_pnl_pct
            total_short_periods += short_periods

            ts_df.loc["NumOfLongTrades", ticker] = int(len(long_periods))
            ts_df.loc["NumOfShortTrades", ticker] = int(len(short_periods))
            long_win_trades = len([1 for pnl_pct in long_pnl_pct if (pnl_pct>0)])
            short_win_trades = len([1 for pnl_pct in short_pnl_pct if (pnl_pct>0)])

            ts_df.loc["WinRateOfTrades(%)", ticker] = "{:.1f}%".format(100*(long_win_trades+short_win_trades)/(len(long_pnl_pct)+len(short_pnl_pct))) if (len(long_pnl_pct)+len(short_pnl_pct) != 0) else "NA"
            ts_df.loc["WinRateGivenLong(%)", ticker] = "{:.1f}%".format(100*(long_win_trades)/(len(long_pnl_pct))) if (len(long_pnl_pct) != 0) else "NA"
            ts_df.loc["WinRateGivenShort(%)", ticker] = "{:.1f}%".format(100*(short_win_trades)/(len(short_pnl_pct))) if (len(short_pnl_pct) != 0) else "NA"
            combined_pnl = long_pnl_pct + short_pnl_pct
            winning_trades = [pnl for pnl in combined_pnl if (pnl > 0)]
            losing_trades = [pnl for pnl in combined_pnl if (pnl < 0)]
            ts_df.loc["AveragePnlPctPerTrade(%)", ticker] = "{:.3f}%".format(100*sum(combined_pnl)/len(combined_pnl)) if (len(combined_pnl) != 0) else "NA"
            ts_df.loc["AveragePnlPctGivenLong(%)", ticker] = "{:.3f}%".format(100*sum(long_pnl_pct)/len(long_pnl_pct)) if (len(long_pnl_pct) != 0) else "NA"
            ts_df.loc["AveragePnlPctGivenShort(%)", ticker] = "{:.3f}%".format(100*sum(short_pnl_pct)/len(short_pnl_pct)) if (len(short_pnl_pct) != 0) else "NA"
            ts_df.loc["AveragePnlPctGivenWin(%)", ticker] = "{:.3f}%".format(100*sum(winning_trades)/len(winning_trades)) if (len(winning_trades) != 0) else "NA"
            ts_df.loc["AveragePnlPctGivenLose(%)", ticker] = "{:.3f}%".format(100*sum(losing_trades)/len(losing_trades)) if (len(losing_trades) != 0) else "NA"
            combined_periods = long_periods + short_periods
            ts_df.loc["AverageHoldingPeriodPerTrade(Bar)", ticker] = "{:.1f}".format(sum(combined_periods)/len(combined_periods)) if (len(combined_periods) != 0) else "NA"
            ts_df.loc["AverageHoldingPeriodGivenLong(Bar)", ticker] = "{:.1f}".format(sum(long_periods)/len(long_periods)) if (len(long_periods) != 0) else "NA"
            ts_df.loc["AverageHoldingPeriodGivenShort(Bar)", ticker] = "{:.1f}".format(sum(short_periods)/len(short_periods)) if (len(short_periods) != 0) else "NA"
            pvalue = self.get_pvalue(combined_pnl)
            ts_df.loc["p-value", ticker] = "{:.4f}%".format(pvalue*100) if pvalue != "NA" else pvalue

        ts_df.loc["NumOfTotalTrades", "Total"] = int(len(total_long_periods))+int(len(total_short_periods))
        ts_df.loc["NumOfLongTrades", "Total"] = int(len(total_long_periods))
        ts_df.loc["NumOfShortTrades", "Total"] = int(len(total_short_periods))
        long_win_trades = len([1 for pnl_pct in total_long_pnl_pct if (pnl_pct>0)])
        short_win_trades = len([1 for pnl_pct in total_short_pnl_pct if (pnl_pct>0)])

        ts_df.loc["WinRateOfTrades(%)", "Total"] = "{:.1f}%".format(100*(long_win_trades+short_win_trades)/(len(total_long_pnl_pct)+len(total_short_pnl_pct))) if (len(total_long_pnl_pct)+len(total_short_pnl_pct) != 0) else "NA"
        ts_df.loc["WinRateGivenLong(%)", "Total"] = "{:.1f}%".format(100*(long_win_trades)/(len(total_long_pnl_pct))) if (len(total_long_pnl_pct) != 0) else "NA"
        ts_df.loc["WinRateGivenShort(%)", "Total"] = "{:.1f}%".format(100*(short_win_trades)/(len(total_short_pnl_pct))) if (len(total_short_pnl_pct) != 0) else "NA"
        combined_pnl = total_long_pnl_pct + total_short_pnl_pct
        winning_trades = [pnl for pnl in combined_pnl if (pnl > 0)]
        losing_trades = [pnl for pnl in combined_pnl if (pnl < 0)]
        ts_df.loc["AveragePnlPctPerTrade(%)", "Total"] = "{:.3f}%".format(100*sum(combined_pnl)/len(combined_pnl)) if (len(combined_pnl) != 0) else "NA"
        ts_df.loc["AveragePnlPctGivenLong(%)", "Total"] = "{:.3f}%".format(100*sum(total_long_pnl_pct)/len(total_long_pnl_pct)) if (len(total_long_pnl_pct) != 0) else "NA"
        ts_df.loc["AveragePnlPctGivenShort(%)", "Total"] = "{:.3f}%".format(100*sum(total_short_pnl_pct)/len(total_short_pnl_pct)) if (len(total_short_pnl_pct) != 0) else "NA"
        ts_df.loc["AveragePnlPctGivenWin(%)", "Total"] = "{:.3f}%".format(100*sum(winning_trades)/len(winning_trades)) if (len(winning_trades) != 0) else "NA"
        ts_df.loc["AveragePnlPctGivenLose(%)", "Total"] = "{:.3f}%".format(100*sum(losing_trades)/len(losing_trades)) if (len(losing_trades) != 0) else "NA"
        combined_periods = total_long_periods + total_short_periods
        ts_df.loc["AverageHoldingPeriodPerTrade(Bar)", "Total"] = "{:.1f}".format(sum(combined_periods)/len(combined_periods)) if (len(combined_periods) != 0) else "NA"
        ts_df.loc["AverageHoldingPeriodGivenLong(Bar)", "Total"] = "{:.1f}".format(sum(total_long_periods)/len(total_long_periods)) if (len(total_long_periods) != 0) else "NA"
        ts_df.loc["AverageHoldingPeriodGivenShort(Bar)", "Total"] = "{:.1f}".format(sum(total_short_periods)/len(total_short_periods)) if (len(total_short_periods) != 0) else "NA"
        pvalue = self.get_pvalue(combined_pnl)
        ts_df.loc["p-value", "Total"] = "{:.4f}%".format(pvalue*100) if pvalue != "NA" else pvalue
        return ts_df 

    def get_pvalue(self, total_pnl_pct_lst):
        if (len(total_pnl_pct_lst) < 2):
            return "NA"
        
        # first get z_score
        average_return = sum(total_pnl_pct_lst)/len(total_pnl_pct_lst)
        stddev = np.std(total_pnl_pct_lst, ddof=1)/pow(len(total_pnl_pct_lst),0.5)
        zscore = abs(average_return)/stddev
        pvalue = 1 - norm.cdf(zscore)

        return pvalue


                    




            