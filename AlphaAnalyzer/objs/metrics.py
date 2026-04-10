import pandas as pd
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None
import os 
import platform
from objs.trade_statistics import TradeStatistics
from tqdm import tqdm
import multiprocessing
import seaborn as sns


class Metrics:
    def __init__(self, filename, oos_date = "2020-01-01", risk_free_rate = 0.03, beta_series_file = "csv\\STI_index.csv", print_out = True, remove_initial_stagnent = True):
        #print(filename)
        self.return_df = pd.read_csv(filename, index_col = 0)
        
        try:
            try:
                self.return_df.index = pd.to_datetime(self.return_df.index, format="%Y-%m-%d")
            except:
                self.return_df.index = pd.to_datetime(self.return_df.index, format="%Y-%m-%d %H:%M:%S")
        except:
            self.return_df.index = pd.to_datetime(self.return_df.index, format="%d/%m/%Y")
            
        if (remove_initial_stagnent):
            non_zero_return_df = self.return_df.loc[self.return_df["pnl(%)"] != 0]
            if (non_zero_return_df.shape[0] != 0):
                first_nonstagnant_row = self.return_df.loc[self.return_df["pnl(%)"] != 0].index[0]
                self.return_df = self.return_df.loc[first_nonstagnant_row:]

        self.beta_series_df = pd.read_csv(beta_series_file, index_col = 0)
        self.beta_series_df.index = pd.to_datetime(self.beta_series_df.index)

        self.oos_date = oos_date
        self.risk_free_rate = risk_free_rate
        self.filename = filename.split("\\")[-1].replace(".csv", "")

        if (print_out):
            print(f"Strategy Name: {self.filename}")

        self.pnl_df, self.bars_in_a_year = cluster_pnl_to_daily(pd.DataFrame(self.return_df["pnl(%)"]))


    def run(self):
        # splitting the return df to three series, one is full series, another is in-sample series and last one is out of sample series
        full_series_df = self.pnl_df
        
        is_series_df = full_series_df[full_series_df.index < pd.to_datetime(self.oos_date)]
        oos_series_df = full_series_df[full_series_df.index >= pd.to_datetime(self.oos_date)]
        
        series_dict = {"full":full_series_df, "in-sample":is_series_df, "out-of-sample": oos_series_df} 
        series_result = {}

        ts = TradeStatistics(self.return_df)
        self.ts_df = ts.run()

        for series_name, df in series_dict.items():
            result_table = {}
            start_end_date = self.get_start_end_date(df)
            print(start_end_date, series_name)
            df = self.create_nav_series(df)

            if (series_name == "full"):
                self.full_series_df = df

            cagr = self.get_cagr(df)
            trading_bars = self.get_trading_bars(df)
            volatility = self.get_volatility(df)
            sharpe = self.get_sharpe(cagr, volatility)
            downside_volatility = self.get_downside_volatility(df)
            sortino_ratio = self.get_sortino_ratio(cagr, downside_volatility)
            max_drawdown, max_drawdown_bars, drawdown_duration_adjusted_metric = self.get_max_drawdown_info(df)
            var95, cvar95 = self.get_value_at_risk(df, 0.95)
            market_beta = self.get_correlation(df)
            calmar_ratio = self.get_calmar_ratio(cagr, max_drawdown)
            active_bars_pct = self.get_pct_bars_active(df)

            # trade statistics
            result_table["Period"] = start_end_date
            result_table["CAGR"] = '{:.3f}%'.format(cagr*100)
            result_table["Volatility"] ='{:.3f}%'.format(volatility*100)
            result_table["Risk-Free Rate"] = '{:.2f}%'.format(self.risk_free_rate*100)
            result_table["Sharpe"] = '{:.3f}'.format(sharpe)
            result_table["Downside Volatility"] = '{:.3f}%'.format(downside_volatility*100)
            result_table["Max Drawdown"] = '{:.3f}%'.format(max_drawdown*100)
            result_table["Longest Recovery Bars"] = int(max_drawdown_bars)
            result_table["Drawdown-Duration Adjusted Loss"] = '{:.3f}%'.format(drawdown_duration_adjusted_metric*100)
            result_table["VaR95"] = '{:.3f}%'.format(var95*100)
            result_table["CVaR95"] = '{:.3f}%'.format(cvar95*100)
            result_table["MarketBeta"] = '{:.3f}'.format(market_beta)
            result_table["Sortino Ratio"] = '{:.3f}'.format(sortino_ratio)
            result_table["Calmar Ratio"] = '{:.3f}'.format(calmar_ratio)
            result_table["Trading Bars"] = trading_bars
            result_table["Active Bars Pct"] =  '{:.1f}%'.format(active_bars_pct*100)
            
            series_result[series_name] = result_table

        self.series_result = series_result
        self.result_df = pd.DataFrame(series_result)
        return 


    def display_table(self):
        display(self.result_df)

        if (self.ts_df is not None):
            display(self.ts_df)

        return 

    def plot_nav(self, figsize=(10, 6), log_scale=True):
        # Split IS and OOS periods
        self.is_series_df = self.full_series_df[self.full_series_df.index < pd.to_datetime(self.oos_date)]
        self.oos_series_df = self.full_series_df[self.full_series_df.index >= pd.to_datetime(self.oos_date)]
        
        plt.figure(figsize=figsize)
        
        # Plot in-sample NAV
        plt.plot(self.is_series_df.index, self.is_series_df["nav"], 
                color='black', label="In-sample")
        
        # Plot out-of-sample NAV
        plt.plot(self.oos_series_df.index, self.oos_series_df["nav"], 
                color='blue', label="Out-of-sample")
        
        # Apply log scaling if requested
        if log_scale:
            plt.yscale("log")
        
        # Labels, legend, and formatting
        plt.legend(loc="upper left", fontsize=9)
        plt.xlabel("Date", fontsize=10)
        plt.ylabel("Net Asset Value (NAV)", fontsize=10)
        plt.xticks(fontsize=7)
        plt.title("Cumulative NAV" + (" (Log Scale)" if log_scale else ""), fontsize=12)
        
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.tight_layout()
        plt.show()
        
        return self.full_series_df
    
    def plot_cumuative_returns(self, figsize=(12, 6)):
        
        # Compute cumulative returns from NAV
        self.full_series_df["cum_return"] = self.full_series_df["nav"] / self.full_series_df["nav"].iloc[0] - 1
        self.full_series_df["log_cum_return"] = np.log(self.full_series_df["nav"] / self.full_series_df["nav"].iloc[0])
        
        # Split IS and OOS
        is_series_df = self.full_series_df[self.full_series_df.index < pd.to_datetime(self.oos_date)]
        oos_series_df = self.full_series_df[self.full_series_df.index >= pd.to_datetime(self.oos_date)]


        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

        # --- Plot 1: Cumulative Returns ---
        axes[0].plot(is_series_df.index, is_series_df["cum_return"], color="black", label="In-sample")
        axes[0].plot(oos_series_df.index, oos_series_df["cum_return"], color="blue", label="Out-of-sample")
        axes[0].set_ylabel("Cumulative Return")
        axes[0].legend(loc="upper left")
        axes[0].grid(True)

        # --- Plot 2: Log Cumulative Returns ---
        axes[1].plot(is_series_df.index, is_series_df["log_cum_return"], color="black", label="In-sample")
        axes[1].plot(oos_series_df.index, oos_series_df["log_cum_return"], color="blue", label="Out-of-sample")
        axes[1].set_ylabel("Log Cumulative Return")
        axes[1].set_xlabel("Date")
        axes[1].legend(loc="upper left")
        axes[1].grid(True)

        plt.xticks(fontsize=7)
        plt.tight_layout()
        plt.show()

    def get_start_end_date(self, df):
        print(df.index[0])
        start_date = df.index[0].strftime('%Y-%m-%d')
        end_date = df.index[-1].strftime('%Y-%m-%d')
        start_end_date = start_date + " to " + end_date
        return start_end_date

    def create_nav_series(self, return_df):
        return_df.loc[:, 'nav'] = (1 + return_df.loc[:,'pnl(%)']).cumprod()
        return return_df
    
    def get_latest_nav(self, return_df):
        last_index = return_df.index[-1]
        return return_df.loc[last_index, "nav"]

    def get_cagr(self, return_df):
        years = (return_df.index[-1] - return_df.index[0]).days/365.25
        last_index = return_df.index[-1]
        cagr = (return_df.loc[last_index, "nav"] ** (1/years)) - 1
        return cagr
    
    def get_trading_bars(self, return_df):
        return return_df.shape[0]
    
    def get_volatility(self, return_df):
        vol = np.std(return_df["pnl(%)"], ddof = 1) * np.sqrt(self.bars_in_a_year)
        return vol
    
    def get_sharpe(self, cagr, volatility):
        if (volatility == 0):
            return 0

        return (cagr-self.risk_free_rate)/volatility

    def get_downside_volatility(self, return_df):
        filtered_return_df = return_df[return_df["pnl(%)"] < 0]
        downside_vol = np.std(filtered_return_df["pnl(%)"], ddof = 1) * np.sqrt(self.bars_in_a_year)
        if (np.isnan(downside_vol)):
            return 0

        return downside_vol
    
    def get_sortino_ratio(self, cagr, downside_vol):
        if (downside_vol == 0):
            return 0

        return (cagr-self.risk_free_rate)/downside_vol
    
    def get_max_drawdown(self, return_df):
        return_df.loc[:, "rolling_max_nav"] = return_df.loc[:,"nav"].cummax()
        return_df["rolling_nav_drawdown_from_high"] = return_df["nav"]/return_df["rolling_max_nav"]-1
        return return_df["rolling_nav_drawdown_from_high"].min()
    
    def get_max_drawdown_info(self, return_df):
        current_drawdown_bars = 0
        max_drawdown_bars = 0
        highest_nav_recorded = 0
        nav_col = return_df.columns.get_loc("nav")
        previous_peak = 1
        lowest_nav_since_new_peak = 1
        drawdowns = []
        max_drawdown = 0
        bars_in_drawdown = 0
        bars_in_drawdown_lst = []

        for i in range(return_df.shape[0]):
            current_nav = return_df.iat[i, nav_col]
            highest_nav_recorded = max(highest_nav_recorded, current_nav)
            current_drawdown_bars = current_drawdown_bars + 1 if (current_nav != highest_nav_recorded) else 0
            max_drawdown = min(max_drawdown, current_nav/highest_nav_recorded-1)
            max_drawdown_bars = max(max_drawdown_bars, current_drawdown_bars) 

            if (current_nav >= previous_peak):
                if (lowest_nav_since_new_peak != previous_peak):
                    drawdowns.append(lowest_nav_since_new_peak/previous_peak-1)
                    bars_in_drawdown_lst.append(bars_in_drawdown)
    
                previous_peak = current_nav
                lowest_nav_since_new_peak = current_nav
                bars_in_drawdown = 0
            else:
                bars_in_drawdown += 1
                lowest_nav_since_new_peak = min(lowest_nav_since_new_peak, current_nav)
            
            #last bar
            if (i == return_df.shape[0]-1 and bars_in_drawdown != 0):
                drawdowns.append(lowest_nav_since_new_peak/previous_peak-1)
                bars_in_drawdown_lst.append(bars_in_drawdown)
 
        drawdown_duration_adjusted_metric = 0
        for i in range(len(drawdowns)):
            drawdown_duration_adjusted_metric += ((bars_in_drawdown_lst[i]/return_df.shape[0])*drawdowns[i])

        if (max_drawdown_bars == 0):
            return 0, 0, 0

        return max_drawdown, max_drawdown_bars, drawdown_duration_adjusted_metric

    def get_value_at_risk(self, return_df, percentile):
        var_val = np.percentile(return_df["pnl(%)"], 1-percentile)
        cvar_subset_df = return_df[return_df["pnl(%)"] <= var_val]
        cvar_val = cvar_subset_df["pnl(%)"].mean()
        return var_val, cvar_val

    def get_calmar_ratio(self, cagr, max_dd):
        if (max_dd == 0):
            return 0
        
        return (cagr-self.risk_free_rate)/abs(max_dd)

    def get_pct_bars_active(self, return_df):
        pct_bars_active = len(return_df[return_df['pnl(%)'] != 0]) / return_df.shape[0]
        return pct_bars_active
    
    def get_correlation(self, return_df):
        # other_series = pd.DataFrame(self.beta_series_df)
        # other_series = other_series.rename(columns={"pnl(%)", "pnl2(%)"})
        correlated_val = return_df["pnl(%)"].corr(self.beta_series_df["pnl(%)"])
        return correlated_val
    
def plot_multiple_series(*metrics):
    # metrics become a tuple
    fig, ax = plt.subplots()  
    
    for metric in metrics:
        strategy_name = metric.filename
        oos_date = pd.to_datetime(metric.oos_date)
        in_sample_length = metric.full_series_df[metric.full_series_df.index < oos_date].shape[0]
        
        ax.plot(metric.full_series_df.index, metric.full_series_df["nav"], label = strategy_name)
        
    ax.axvspan(metric.full_series_df.index[in_sample_length], metric.full_series_df.index[-1], color='lightblue', alpha=0.5)
    plt.legend(loc="upper left")
    plt.xlabel("Date")
    plt.ylabel("Net Asset Value (NAV)")
    plt.xticks(fontsize=7)
    plt.show()
    
def plot_multiple_series_v2(*metrics):
    """
    Plots NAV and log-scaled NAV for multiple strategy series.
    Returns the merged DataFrame (outer join on dates).
    """
    # Merge all NAV series via outer join
    merged_df = None
    for metric in metrics:
        series = metric.full_series_df[["nav"]].copy()
        series.columns = [metric.filename]  # Rename to strategy name
        
        if merged_df is None:
            merged_df = series
        else:
            merged_df = merged_df.join(series, how="outer")
    
    # Sort by date
    merged_df.sort_index(inplace=True)

    # --- Plot 1: Cumulative Returns ---
    fig, ax = plt.subplots(figsize=(10, 5))
    for col in merged_df.columns:
        ax.plot(merged_df.index, merged_df[col], label=col)
    ax.set_title("Cumulative Returns")
    ax.set_xlabel("Date")
    ax.set_ylabel("NAV")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(fontsize=7)

    # --- Plot 2: Log-scaled Cumulative Returns ---
    fig, ax = plt.subplots(figsize=(10, 5))
    for col in merged_df.columns:
        ax.plot(merged_df.index, np.log(merged_df[col]), label=col)
    ax.set_title("Log-scaled Cumulative Returns")
    ax.set_xlabel("Date")
    ax.set_ylabel("Log(NAV)")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(fontsize=7)

    plt.show()

    return merged_df


def single_result_eval(folder_path, file_name, oos_date, risk_free_rate, print_out, remove_initial_stagnent, return_dict):
    if (file_name[0] != "_"):
        file_path = os.path.join(folder_path, file_name)
        metrics = Metrics(file_path, oos_date, risk_free_rate, print_out = print_out, remove_initial_stagnent=remove_initial_stagnent)
        metrics.run()

        # process the data
        series_result = metrics.series_result
        dampened_result = {}
        dampened_result["file_name"] = file_name

        for period, result in series_result.items():
            for key_metric, value in result.items():
                dampened_result[f"{period}_{key_metric}"] = value

        return_dict[file_name] = dampened_result
    return 
    # return dampened_result

        
def generate_clustered_result(folder_path, oos_date = "2020-01-01", risk_free_rate = 0.03, print_out = False, remove_initial_stagnent = False):
    metric_clusters = []
    folder_name_start = folder_path.rfind("\\")
    folder_name = folder_path[folder_name_start+1:]
    print(f"folder name start is: {folder_name_start}")
    print(f"folder name is: {folder_name}")
    manager = multiprocessing.Manager()
    print(f"manager is: {manager}")
    return_dict = manager.dict()
    jobs = []
    

    # force 5 at a time
    count = 0
    max_processes = 10

    for file_name in tqdm(os.listdir(folder_path)):
        p = multiprocessing.Process(target=single_result_eval, args=(folder_path, file_name, oos_date, risk_free_rate, print_out, remove_initial_stagnent, return_dict))
        count += 1
        jobs.append(p)
        p.start()

        if (count == max_processes):
            # print("pausing now! until jobs are done")
            for job in jobs:
                job.join()
            # print("batches are done")
            count = 0
            jobs = []
    for key, val in return_dict.items():
        metric_clusters.append(val)
    print(f"return dict is: {return_dict}")
    clustered_df = pd.DataFrame(metric_clusters)
    print(metric_clusters)

    # add rank to it 
    clustered_df = rank_performance_statistics(clustered_df, ranking_cols = ["full_Sharpe", "in-sample_Sharpe", "out-of-sample_Sharpe"])
    clustered_data_path = f"{folder_path}/_clustered_{folder_name}.csv" if (platform.system() == "Linux") else f"{folder_path}\\_clustered_{folder_name}.csv"

    #clustered_df.to_csv(clustered_data_path)

    # for file_name in tqdm(os.listdir(folder_path)):
    #     if (file_name[0] != "_"):
    #         file_path = os.path.join(folder_path, file_name)
    #         # print(file_path)
    #         metrics = Metrics(file_path, oos_date, risk_free_rate, print_out = print_out, remove_initial_stagnent=remove_initial_stagnent)
    #         metrics.run()

    #         # process the data
    #         series_result = metrics.series_result
    #         dampened_result = {}
    #         dampened_result["file_name"] = file_name

    #         for period, result in series_result.items():
    #             for key_metric, value in result.items():
    #                 dampened_result[f"{period}_{key_metric}"] = value

    #         metric_clusters.append(dampened_result)
    
    # clustered_df = pd.DataFrame(metric_clusters)

    # # add rank to it 
    # clustered_df = rank_performance_statistics(clustered_df, ranking_cols = ["full_Sharpe", "in-sample_Sharpe", "out-of-sample_Sharpe"])
    # clustered_data_path = f"{folder_path}/_clustered_{folder_name}.csv" if (platform.system() == "Linux") else f"{folder_path}\\_clustered_{folder_name}.csv"

    # clustered_df.to_csv(clustered_data_path)
    # return
    return clustered_df, clustered_data_path

def rank_performance_statistics(clustered_df, ranking_cols = []):
    for col in ranking_cols:
        clustered_df[col] = clustered_df[col].astype(float)
        clustered_df.insert(loc=clustered_df.columns.get_loc(col)+1, column = col+"_rank", value = clustered_df[col].rank(method='min', ascending=False).values)
    
    return clustered_df

def cluster_pnl_to_daily(df_pnl):
    date_only_return_dict = {}
    pnl_col = df_pnl.columns.get_loc("pnl(%)")
    
    date_only_return_dict[df_pnl.index[0].date()] = df_pnl.iat[0, pnl_col]

    for i in range(1, df_pnl.shape[0]):
        if (df_pnl.index[i].date() != df_pnl.index[i-1].date()):
            date_only_return_dict[df_pnl.index[i].date()] = df_pnl.iat[i, pnl_col]
        else:
            date_only_return_dict[df_pnl.index[i].date()] = (1+date_only_return_dict[df_pnl.index[i].date()])*(1+df_pnl.iat[i, pnl_col])-1
    
    df_pnl_grouped_daily = pd.DataFrame(date_only_return_dict.values(), columns = ["pnl(%)"], index=pd.to_datetime(list(date_only_return_dict.keys())))
    years_between = abs((df_pnl_grouped_daily.index[df_pnl_grouped_daily.shape[0]-1] - df_pnl_grouped_daily.index[0]).days)/365.25
    bars_per_year = round(df_pnl_grouped_daily.shape[0]/years_between)
    
    return df_pnl_grouped_daily, bars_per_year  

#-------------------------------------------------------------------------------------------------------#

def preprocess_multi_series(merged_df: pd.DataFrame):
    '''
    Each column name is the strategy name and values are its NAV
    '''
    # Drop all Nan, ensure that no nan
    merged_df = merged_df.dropna()
    merged_df = merged_df.sort_index(ascending=True)
    
    # Make the engineered columns
    strategies = merged_df.columns
    for strat in strategies:
        merged_df[strat + '_returns'] = (merged_df[strat] - merged_df[strat].shift(1)) / merged_df[strat].shift(1)
    return merged_df.dropna(), list(strategies)


def plot_multi_series_returns(df: pd.DataFrame, strategies: list):
    """
    Given dataframe with NAV columns and *_returns columns, plots:
    1. Cumulative Returns (linear scale)
    2. Cumulative Log Returns (all on one plot)
    3. Density Plot of Periodic Returns (all strategies overlaid)
    Also prints mean, std, skewness, and kurtosis for each strategy.
    """
    plt.style.use("seaborn-v0_8-darkgrid")

    # --- Summary Statistics ---
    stats_records = []
    for strat in strategies:
        r = df[strat + "_returns"].dropna()
        stats_records.append({
            "Strategy": strat,
            "Mean":     r.mean(),
            "Std Dev":  r.std(),
            "Skewness": r.skew(),
            "Kurtosis": r.kurtosis(),  # excess kurtosis (normal = 0)
        })

    stats_df = pd.DataFrame(stats_records).set_index("Strategy")
    print("\n===== Periodic Return Statistics =====")
    display(stats_df)
    print("======================================\n")

    # --- Plots ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 16))

    # 1. Cumulative Returns
    for strat in strategies:
        cum_ret = (1 + df[strat + "_returns"]).cumprod()
        axes[0].plot(df.index, cum_ret, label=strat)
    axes[0].set_title("Cumulative Returns", fontsize=14)
    axes[0].set_ylabel("Growth of $1")
    axes[0].legend()

    # 2. Cumulative Log Returns
    for strat in strategies:
        log_cum_ret = np.log1p(df[strat + "_returns"]).cumsum()
        axes[1].plot(df.index, log_cum_ret, label=strat)
    axes[1].set_title("Cumulative Log Returns", fontsize=14)
    axes[1].set_ylabel("Log Return")
    axes[1].legend()

    # 3. Density Plot of Returns
    for strat in strategies:
        sns.kdeplot(df[strat + "_returns"], label=strat, fill=True, alpha=0.3, ax=axes[2])
    axes[2].set_title("Distribution of Periodic Returns", fontsize=14)
    axes[2].set_xlabel("Return")
    axes[2].legend()

    plt.tight_layout()
    plt.show()



def plot_yearly_return_comparison(df: pd.DataFrame, strategies: list):
    """
    Groups returns by calendar year and plots a grouped barplot of total yearly returns.
    """
    plt.style.use("seaborn-v0_8-darkgrid")
    
    yearly_returns = []
    for strat in strategies:
        yearly = df[strat + "_returns"].groupby(df.index.year).apply(lambda r: (1 + r).prod() - 1)
        yearly_returns.append(yearly.rename(strat))
    
    yearly_df = pd.concat(yearly_returns, axis=1)
    yearly_df = yearly_df.reset_index().melt(id_vars="index", var_name="Strategy", value_name="Yearly Return")
    yearly_df.rename(columns={"index": "Year"}, inplace=True)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=yearly_df, x="Year", y="Yearly Return", hue="Strategy")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("Yearly Return Comparison", fontsize=14)
    plt.ylabel("Yearly Return")
    plt.xlabel("Year")
    plt.legend(title="Strategy")
    plt.tight_layout()
    plt.show()
    


def ema_smooth(series: pd.Series, span: int = 20) -> pd.Series:
    """
    Light EMA smoothing to reduce jagged noise in rolling metrics
    without distorting structure.
    """
    return series.ewm(span=span, adjust=False).mean()



def plot_multi_series_underwater_plot(df: pd.DataFrame, strategies: list):
    """
    Plots the underwater plot (drawdowns) for multiple strategies.
    """
    plt.figure(figsize=(12, 6))
    for strat in strategies[::-1]:
        cum_nav = (1 + df[f"{strat}_returns"]).cumprod()
        cum_max = cum_nav.cummax()
        drawdown = (cum_nav / cum_max) - 1
        plt.plot(drawdown.index, drawdown, label=strat)
    plt.title("Underwater Plot (Drawdowns)")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()


def plot_multi_series_rolling_sharpe(df: pd.DataFrame, strategies: list, 
                                                        rf_rate: float = 0.01, 
                                                        lookback_period=6,
                                                        smooth_span: int = 20):
    """
    Rolling Sharpe Ratio plot for multiple strategies.
    lookback_period: number of months.
    """
    monthly_rf = rf_rate / 12
    plt.figure(figsize=(12, 6))
    for strat in strategies[::-1]:
        rolling_sharpe = (
            (df[f"{strat}_returns"].rolling(lookback_period).mean() - monthly_rf) /
            df[f"{strat}_returns"].rolling(lookback_period).std()
        )
        
        rolling_sharpe_smoothed = ema_smooth(rolling_sharpe, span=smooth_span)
        #plt.plot(rolling_sharpe.index, rolling_sharpe, label=strat)
        plt.plot(rolling_sharpe_smoothed.index, rolling_sharpe_smoothed, label=strat)
        
    plt.title(f"Rolling {lookback_period}-Month Sharpe Ratio")
    plt.xlabel("Date")
    plt.ylabel("Sharpe Ratio")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()


def plot_multi_series_rolling_volatility(df: pd.DataFrame, strategies: list, lookback_period=6, smooth_span: int = 20):
    """
    Rolling volatility plot for multiple strategies.
    lookback_period: number of months.
    """
    plt.figure(figsize=(12, 6))
    for strat in strategies[::-1]:
        rolling_vol = df[f"{strat}_returns"].rolling(lookback_period).std() * np.sqrt(12)  # Annualized
        rolling_vol_smoothed = ema_smooth(rolling_vol, span=smooth_span)
        plt.plot(rolling_vol_smoothed.index, rolling_vol_smoothed, label=strat)    
        
        #plt.plot(rolling_vol.index, rolling_vol, label=strat)
    plt.title(f"Rolling {lookback_period}-Month Volatility (Annualized)")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()


def plot_multi_series_rolling_beta_benchmark(df: pd.DataFrame, strategies: list, benchmark_strategy: str, 
                                             lookback_period=6, smooth_span: int = 20):
    """
    Rolling beta of each strategy against a benchmark strategy.
    """
    plt.figure(figsize=(12, 6))
    benchmark_returns = df[f"{benchmark_strategy}_returns"]
    for strat in strategies[::-1]:
        if strat == benchmark_strategy:
            continue
        rolling_beta = df[f"{strat}_returns"].rolling(lookback_period).cov(benchmark_returns) / \
                       benchmark_returns.rolling(lookback_period).var()
        # plt.plot(rolling_beta.index, rolling_beta, label=f"{strat} vs {benchmark_strategy}")
        
        rolling_beta_smoothed = ema_smooth(rolling_beta, span=smooth_span)
        plt.plot(
            rolling_beta_smoothed.index,
            rolling_beta_smoothed,
            label=f"{strat} vs {benchmark_strategy}"
        )
        
        
    plt.title(f"Rolling {lookback_period}-Month Beta vs {benchmark_strategy}")
    plt.xlabel("Date")
    plt.ylabel("Beta")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()






    
