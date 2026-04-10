from matplotlib.pylab import f
import pandas as pd
from datetime import datetime as dt
import yfinance as yf
from objs.stock import Stock
import numpy as np
from src.log import Log
from objs.stock import initialize_stock
from src.postgresql_connector import PostgreSQLConnector
from objs.dividend_history import DividendHistory
import time
import requests
import os

class DataHandler:
    def __init__(self, tickers, start_date = "2000-01-01", end_date = dt.now(), interval = '1d', reference_ticker = "^STI",
                 method = "yfinance", missing_data_handle = "GetPrevious", log = None, adjust_dividend = True):

        print(f'Current OS Directory for Data Handler: {os.getcwd()}')
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.reference_ticker = reference_ticker
        self.method = method
        self.missing_data_handle = missing_data_handle
        self.log = log
        self.adjust_dividend = adjust_dividend

        self.log.download_progress(self.__class__.__name__, "start", method)
        if (self.method == "yfinance"):
            yfinance_class = YFinance()
            self.stock_data, self.reference_data = yfinance_class.download(self.tickers, start_date = self.start_date, end_date = self.end_date,
                                                                           interval = self.interval, reference_ticker = self.reference_ticker, 
                                                                           adjust_dividend = adjust_dividend)
    
            
        elif (self.method == "postgresql"):
            postgresql_class = PostgreSqlDb()
            
            self.stock_data, self.reference_data = postgresql_class.download(self.tickers, start_date = self.start_date, end_date = self.end_date,
                                                                           interval = self.interval, reference_ticker = self.reference_ticker, adjust_dividend = adjust_dividend)
            self.dividend_data = postgresql_class.pull_dividend_history(self.tickers)
            self.adjustment_factor = postgresql_class.pull_adjustment_factor_history(self.tickers)

            for ticker in self.stock_data.keys():
                price_adjustment = PriceAdjustment(self.stock_data[ticker], self.dividend_data[ticker], self.adjustment_factor[ticker])
                self.stock_data[ticker] = price_adjustment.adjust()
        
        elif (self.method == "parquet"):
            parquet_class = Parquet()
            self.stock_data, self.reference_data = parquet_class.download(self.tickers, start_date = self.start_date, end_date = self.end_date,
                                                                           interval = self.interval, reference_ticker = self.reference_ticker,
                                                                           adjust_dividend = adjust_dividend,
                                                                           file_path=f'BackLab/data/daily_ohlcv_448_v2.parquet')
        
        
        # handle data with different time length (especially reference data, since by design it is supposed to be more comprehensive)
        self.get_earliest_start_date()
        self.get_latest_end_date()

        for ticker, data in self.stock_data.copy().items():
            data = data.loc[pd.to_datetime(self.earliest_start_date):pd.to_datetime(self.latest_end_date)]
            self.stock_data[ticker] = data
        
        for ticker, data in self.reference_data.copy().items():
            data = data.loc[pd.to_datetime(self.earliest_start_date):pd.to_datetime(self.latest_end_date)]
            self.reference_data[ticker] = data

        # initialize the stocks
        self.stocks = initialize_stock(tickers)
        self.log.download_progress(self.__class__.__name__, "end", method)
        return 
    
    def get_earliest_start_date(self):
        self.earliest_start_date = dt.now()

        for ticker, data in self.stock_data.copy().items():
            first_date = data.index[0]
            self.earliest_start_date = min(self.earliest_start_date, first_date)

    def get_latest_end_date(self):
        self.latest_end_date = pd.to_datetime("2000-01-01")

        for ticker, data in self.stock_data.copy().items():
            last_date = data.index[-1]
            self.latest_end_date = max(self.latest_end_date, last_date)

    def update_price(self, stocks, date_time, on_bar_prices, all_stock_data):
        '''
        on_bar_prices [list] = a list of prices to update, e.g., update on open only: ["Open"]; update on close/high/low: ["Close", "High", "Low"]
        '''
        for ticker, o in stocks.items():
            stock_data = all_stock_data[ticker]
            for bar_component in on_bar_prices:
                try:
                    bar_component_price = float(stock_data.loc[date_time, bar_component])
                    # price is not nan, we set stock to active
                    if (not o.is_active and not np.isnan(bar_component_price)):
                        o.is_active = True
                    
                    # price component is active but the price is not nan
                    if (o.is_active and np.isnan(bar_component_price)):
                        raise Exception()

                except Exception as err:
                    bar_component_price = MissingPriceHandler.decision(self.log, self.missing_data_handle, err, self.__class__.__name__, bar_component, ticker, date_time, o)

                if (bar_component == "Open"):
                    o.open.add(round(bar_component_price, 6))
                    o.date.add(date_time)

                if (bar_component == "Close"):
                    o.close.add(round(bar_component_price,6))

                if (bar_component == "Volume"):
                    o.volume.add(round(bar_component_price,6))

                if (bar_component == "High"):
                    o.high.add(round(bar_component_price,6))

                    if (o.highest_price is None):
                        o.highest_price = round(bar_component_price,6)
                    else:
                        if (not np.isnan(bar_component_price)):
                            o.highest_price = max(o.highest_price, round(bar_component_price,6))

                if (bar_component == "Low"):
                    o.low.add(round(bar_component_price,6))

                    if (o.lowest_price is None):
                        o.lowest_price = round(bar_component_price,6)
                    else:
                        if (not np.isnan(bar_component_price)):
                            o.lowest_price = min(o.lowest_price, round(bar_component_price,6))
        return stocks

class MissingPriceHandler:
    @staticmethod
    def decision(log, decision_type, err, class_name, bar_component, ticker, date_time, stock):
        if (decision_type == "Raise"):
            raise Exception(f"{err} [{class_name}] Error loading bar component {bar_component} for ticker {ticker} on date time: {date_time}.")
        
        elif (decision_type == "GetPrevious"):
            log.alert_missing_data_point(class_name, ticker, date_time, bar_component)
            if (bar_component == "Open"):
                if (stock.open.length() == 0):
                    return np.nan
                else:
                    return stock.open[0]
            elif (bar_component == "Close"):
                if (stock.close.length() == 0):
                    return np.nan
                else:
                    return stock.close[0]
            elif (bar_component == "High"):
                if (stock.high.length() == 0):
                    return np.nan
                else:
                    return stock.high[0]
            elif (bar_component == "Low"):
                if (stock.low.length() == 0):
                    return np.nan
                else:
                    return stock.low[0]
            elif (bar_component == "Volume"):
                if (stock.volume.length() == 0):
                    return np.nan
                else:
                    return stock.volume[0]

class YFinance:
    def download(self, tickers, start_date, end_date, interval, reference_ticker, adjust_dividend):
        all_stock_data = {}

        #yfinance_data = yf.download(tickers, start = start_date, end = end_date, interval = interval, progress = False)
        yfinance_data = yf.download(tickers, start = start_date, end = end_date, 
                                    interval = interval, progress = False)

        if (len(tickers) > 1):
            for ticker in tickers:
                all_stock_data[ticker] = pd.DataFrame(index = yfinance_data.index)
                for col in yfinance_data.columns:
                    # col => ("Bar Attribute (e.g. close, open, high, low), Ticker")
                    if (col[1] == ticker):
                        all_stock_data[ticker][col[0]] = yfinance_data[col[0]][ticker]
        else:
            all_stock_data[tickers[0]] = yfinance_data

        reference_data = {}
        reference_data[reference_ticker] = yf.download(reference_ticker, start = start_date, end = end_date, interval = interval, progress = False)
        
        all_stock_data, reference_data = self.remove_adj_close(all_stock_data, reference_data, adjust_dividend)

        # sort index, just in case
        for ticker, data in all_stock_data.copy().items():
            all_stock_data[ticker] = data.sort_index()
            assert data.isna().sum().sum() == 0, data.isna().sum()

        print(all_stock_data[tickers[0]].head())
        print(all_stock_data[tickers[0]].columns)
        return all_stock_data, reference_data
    
    @staticmethod
    def remove_adj_close(all_stock_data, reference_data, adjust_dividend):
        copy_stock_data = all_stock_data.copy()
        copy_reference_data = reference_data.copy()
        
        for ticker, df in copy_stock_data.items():
            if 'Adj Close' in df.columns:
                if (adjust_dividend):
                    df["Open"] = (df["Adj Close"]/df["Close"])*df["Open"]
                    df["High"] = (df["Adj Close"]/df["Close"])*df["High"]
                    df["Low"] = (df["Adj Close"]/df["Close"])*df["Low"]
                    df["Close"] = df["Adj Close"]
                df = df.drop("Adj Close", axis = 1)
            all_stock_data[ticker] = df

        for ticker, df in copy_reference_data.items():
            if 'Adj Close' in df.columns:
                if (adjust_dividend):
                    df["Open"] = (df["Adj Close"]/df["Close"])*df["Open"]
                    df["High"] = (df["Adj Close"]/df["Close"])*df["High"]
                    df["Low"] = (df["Adj Close"]/df["Close"])*df["Low"]
                    df["Close"] = df["Adj Close"]
                df = df.drop("Adj Close", axis = 1)
            reference_data[ticker] = df
        return all_stock_data, reference_data

class PostgreSqlDb:
    def download(self, tickers, start_date, end_date, interval, reference_ticker, adjust_dividend):
        all_stock_data = {}
        PG = PostgreSQLConnector("stock")
        
        for ticker in tickers:
            read_query = f"SELECT date, open, high, low, close, volume from ohlcv.{interval} "\
                            f"WHERE ric = '{ticker}' "\
                            "ORDER BY date ASC;"
            
            stock_data = PG.read(read_query, return_df = True)
            stock_data = stock_data.rename(columns={"date": "Date", "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume" : "Volume"})
            stock_data = stock_data.set_index("Date")
            stock_data = stock_data.astype(float)
            stock_data = stock_data.loc[pd.to_datetime(start_date, format="%Y-%m-%d"):pd.to_datetime(end_date, format="%Y-%m-%d")]
            all_stock_data[ticker] = stock_data
            
        reference_data = {}
        read_query = f"SELECT date, open, high, low, close, volume from ohlcv.{interval} "\
                        f"WHERE ric = '{reference_ticker}' "\
                        "ORDER BY date ASC;"
            
        stock_data = PG.read(read_query, return_df = True)
        stock_data = stock_data.rename(columns={"date": "Date", "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume" : "Volume"})
        stock_data = stock_data.set_index("Date")
        stock_data = stock_data.astype(float)
        stock_data = stock_data.loc[pd.to_datetime(start_date, format="%Y-%m-%d"):pd.to_datetime(end_date, format="%Y-%m-%d")]
        reference_data[reference_ticker] = stock_data 

        return all_stock_data, reference_data
    
    def pull_dividend_history(self, tickers):
        PG = PostgreSQLConnector("stock")
        dividend_data = {}
        for ticker in tickers:
            read_query = "SELECT dividend_ex_date, entry_no, dividend_amt from dividend.history "\
                        f"WHERE ric = '{ticker}' "\
                        "ORDER BY dividend_ex_date ASC;"

            dividend_df = PG.read(read_query, return_df = True)
            dividend_df["dividend_amt"] = dividend_df["dividend_amt"].astype(float)
            dividend_df = dividend_df.set_index("dividend_ex_date")
            dividend_data[ticker] = dividend_df

        return dividend_data

    def pull_adjustment_factor_history(self, tickers):
        PG = PostgreSQLConnector("stock")
        adjustment_factor_data = {}
        for ticker in tickers:
            read_query = "SELECT adjustment_factor_ex_date, entry_no, adjustment_factor from adjustment_factor.history "\
                        f"WHERE ric = '{ticker}' "\
                        "ORDER BY adjustment_factor_ex_date ASC;"
            
            adjustment_factor_df = PG.read(read_query, return_df = True)
            adjustment_factor_df["adjustment_factor"] = adjustment_factor_df["adjustment_factor"].astype(float)
            adjustment_factor_df = adjustment_factor_df.set_index("adjustment_factor_ex_date")
            adjustment_factor_data[ticker] = adjustment_factor_df
            
        return adjustment_factor_data
    
    
    
class Parquet:
    ### TO DO ###
    def download(self, tickers, start_date, end_date, interval, reference_ticker, adjust_dividend, file_path=f'fyp-kiat/BackLab/data/daily_ohlcv_448_v2.parquet'):
        
        # Read in the file path that has all the tickers then do the filtering based on the tickers start and end date
        all_stock_data = {}
        reference_data = {}
        full_data = pd.read_parquet(file_path)
        for ticker in tickers:
            stock_data = full_data[full_data['Ticker'] == ticker].drop(columns=['Ticker'])
            stock_data = stock_data.set_index("Date")
            stock_data = stock_data.astype(float)
            stock_data = stock_data.loc[pd.to_datetime(start_date, format="%Y-%m-%d"):pd.to_datetime(end_date, format="%Y-%m-%d")]
            all_stock_data[ticker] = stock_data
            
    
        reference_data = {}
        reference_data[reference_ticker] = yf.download(reference_ticker, start = start_date, end = end_date, interval = interval, progress = False)
        all_stock_data, reference_data = YFinance.remove_adj_close(all_stock_data, reference_data, adjust_dividend)
        return all_stock_data, reference_data
class PriceAdjustment:
    def __init__(self, stock_data: pd.DataFrame, dividend_data: pd.DataFrame, adjustment_factor: pd.DataFrame) -> pd.DataFrame:
        """
        stock_data: ohlcv data
        dividend_data: dividend_ex_date, entry_no, dividend_amt
        adjustment_factor: adjustment_factor_ex_date, entry_no, adjustment_factor
        # only adjust beginning of the day (if dealing with intraday data)
        """
        self.single_stock_data = pd.DataFrame(stock_data).copy(deep=True)
        self.price_adjusted = pd.DataFrame(stock_data).copy(deep=True)
        self.new_dividend_df = self.mark_first_instance(stock_data, dividend_data)
        self.new_adjustment_factor_df = self.mark_first_instance(stock_data, adjustment_factor)
        

    def mark_first_instance(self, stock_data, adjustment_df):
        adjustment_df["first_instance_date"] = ""
        for i in range(adjustment_df.shape[0]):
            if (adjustment_df.index[i] >= stock_data.index[0]):
                # marker
                temp = stock_data[(stock_data.index >= adjustment_df.index[i])].index
                if (len(temp)!=0):
                    adjustment_df.iat[i, adjustment_df.columns.get_loc("first_instance_date")] = temp[0]
        adjustment_df = adjustment_df[adjustment_df["first_instance_date"] != ""]
        return adjustment_df                


    def adjust(self):
        col_index = {"Open": self.price_adjusted.columns.get_loc("Open"), "Close": self.price_adjusted.columns.get_loc("Close"),
                      "High": self.price_adjusted.columns.get_loc("High"), "Low": self.price_adjusted.columns.get_loc("Low"),
                      "Volume": self.price_adjusted.columns.get_loc("Volume")}
        
        if (self.new_dividend_df.shape[0] == 0 and self.new_adjustment_factor_df.shape[0] == 0):
            return self.price_adjusted

        div_first_instance_list = self.new_dividend_df["first_instance_date"].values.tolist()
        adjustment_factor_first_instance_list = self.new_adjustment_factor_df["first_instance_date"].values.tolist()

        for i in range(self.price_adjusted.shape[0]):   
            if (i > 0):
                latest_open_price = float(self.single_stock_data.iat[i, col_index["Open"]])
                latest_vol_change = self.single_stock_data.iat[i, col_index["Volume"]]/self.single_stock_data.iat[i-1, col_index["Volume"]]
                if (self.price_adjusted.index[i] in div_first_instance_list):
                    div_amt = self.new_dividend_df[self.new_dividend_df["first_instance_date"] == self.price_adjusted.index[i]].loc[:, "dividend_amt"].sum()
                    latest_open_price = latest_open_price + div_amt
                    
                if (self.price_adjusted.index[i] in adjustment_factor_first_instance_list):
                    adjustment_factor = self.new_adjustment_factor_df[self.new_adjustment_factor_df["first_instance_date"] == self.price_adjusted.index[i]].loc[:, "adjustment_factor"].prod()
                    latest_open_price = latest_open_price * (1/adjustment_factor)
                    latest_vol_change = (self.single_stock_data.iat[i, col_index["Volume"]]*adjustment_factor)/self.single_stock_data.iat[i-1, col_index["Volume"]]
                
                # adjust open high low close
                new_open_price = (latest_open_price / self.single_stock_data.iat[i-1, col_index["Close"]]) * self.price_adjusted.iat[i-1, col_index["Close"]]
                self.price_adjusted.iat[i, col_index["Open"]] = new_open_price
                self.price_adjusted.iat[i, col_index["High"]] = (self.single_stock_data.iat[i, col_index["High"]]/self.single_stock_data.iat[i, col_index["Open"]])*new_open_price
                self.price_adjusted.iat[i, col_index["Low"]] = (self.single_stock_data.iat[i, col_index["Low"]]/self.single_stock_data.iat[i, col_index["Open"]])*new_open_price
                self.price_adjusted.iat[i, col_index["Close"]] = (self.single_stock_data.iat[i, col_index["Close"]]/self.single_stock_data.iat[i, col_index["Open"]])*new_open_price
                self.price_adjusted.iat[i, col_index["Volume"]] = latest_vol_change*self.price_adjusted.iat[i-1, col_index["Volume"]]

                # print(f"datetime: {self.price_adjusted.index[i]}, adj open price: {self.price_adjusted.iat[i, col_index['Open']]}, adj close: {self.price_adjusted.iat[i, col_index['Close']] }")

        price_adjustment_factor = self.single_stock_data.iat[self.price_adjusted.shape[0]-1, col_index["Close"]]/self.price_adjusted.iat[self.price_adjusted.shape[0]-1, col_index["Close"]]
        volume_adjustment_factor = self.single_stock_data.iat[self.price_adjusted.shape[0]-1, col_index["Volume"]]/self.price_adjusted.iat[self.price_adjusted.shape[0]-1, col_index["Volume"]]

        self.price_adjusted[["Open", "High", "Low", "Close"]] = self.price_adjusted[["Open", "High", "Low", "Close"]] * price_adjustment_factor
        self.price_adjusted["Volume"] = self.price_adjusted["Volume"] * volume_adjustment_factor

        return self.price_adjusted
        
# testing
if __name__ == "__main__":

    tickers = ["D05.SI", "S51.SI"]
    data_handler = DataHandler(tickers)
    

        
            
        