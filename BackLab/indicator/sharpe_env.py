#import indicator_prerun
import numpy as np
from series import Series
from indicator_validity import indicator_validity
import pandas as pd
from collections import deque
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import minimize
import pandas_ta as ta
from scipy.stats import zscore
import random

class ShapeEnv:
    def __init__(self, sector_data, tickers, rsi_to_ticker, ticker_to_rsi, lookback=1, sector_weights=None, gross_proportion=2,
                 rebalance_interval=1):
        np.random.seed(42)
        random.seed(42)
        self.active_bars = 0
        self.values = Series()
        self.sector_data = sector_data

        # Strategy Parameters
        self.rankings = {i: None for i in tickers}
        self.sector_weights = {i:1 for i in self.sector_data.columns if i != 'Date' and i != 'Ticker'} if sector_weights is None else sector_weights
        self.lookback = lookback
        self.gross_proportion = gross_proportion
        self.rebalance_interval = self.lookback if rebalance_interval == 0 else rebalance_interval
        # Lookback period means how long we should lookback to look at the factors for ranking

        # Mappings
        self.tickers = tickers
        self.ticker_rsi = [ticker_to_rsi[i] for i in tickers]
        self.rsi_to_ticker = rsi_to_ticker
        self.ticker_to_rsi = ticker_to_rsi

        # Dictionary for storing returns of each stock, stored in ric
        self.deque_returns = {self.ticker_to_rsi[i]: deque([]) for i in tickers}

        # Maintain a Close Dataframe, each time we add it in the update function
        self.open_prices = {self.ticker_to_rsi[i]: deque([]) for i in tickers}

    def __getitem__(self, index):
        return self.values[index]
    
    def __str__(self):
        return str(self.values)
    
    def __repr__(self):
        return repr(self.values)

    def length(self):
        return self.values.length()
    
    '''Start of Environment Functions'''
    def portfolio(self, returns, weights):
        weights = np.array(weights)
        rets = returns.mean() * 252
        covs = returns.cov() * 252
        P_ret = np.sum(rets * weights)
        P_vol = np.sqrt(np.dot(weights.T, np.dot(covs, weights)))
        P_sharpe = P_ret / P_vol
        return np.array([P_ret, P_vol, P_sharpe])
    
    def preprocess_state(self, state):
        # Making covariance and reshaping
        #print(f"State Shape before cov: {pd.DataFrame(state).shape}")
        cov_state = pd.DataFrame(state).cov().values
        #print(f"State Shape after cov: {cov_state.shape}")
        if np.isnan(cov_state).any() or cov_state.shape == (0, 0):
            print(state)
        return cov_state  
    
    # Improve this 
    def get_state(self, t, lookback):
        assert lookback <= t
        start_idx, end_idx = t-lookback, t
        #print(f"Start: {start_idx}, End: {end_idx}, Active Bars: {self.active_bars}")

        #print(len(self.deque_returns[self.ticker_rsi[0]]))
        
        # Form a covariance matrix of returns for each stock
        state = {}
        for rsi in self.ticker_rsi:
            #print(f"RSI: {rsi}, length of returns: {len(self.deque_returns[rsi])}")
            state[rsi] = np.array([self.deque_returns[rsi][i] for i in range(start_idx, end_idx)])
            assert len(state[rsi]) == lookback, f"Length of State: {len(state[rsi])} != Lookback: {lookback}"
        
        return self.preprocess_state(state)
        # Make it into np.array
        

    
    def get_reward(self, action, action_t, reward_t, alpha = 0.01, is_eval=False):

        def local_portfolio(returns, weights):
            weights = np.array(weights)
            rets = returns.mean() # * 252
            covs = returns.cov() # * 252
            P_ret = np.sum(rets * weights)
            P_vol = np.sqrt(np.dot(weights.T, np.dot(covs, weights)))
            P_sharpe = P_ret / P_vol
            return np.array([P_ret, P_vol, P_sharpe])
        
        data_period = pd.DataFrame({rsi: np.array([self.open_prices[rsi][i] for i in range(action_t, reward_t)]) for rsi in self.ticker_rsi})
        weights = action
        returns =  pd.DataFrame({rsi: np.array([self.deque_returns[rsi][i] for i in range(action_t, reward_t)]) for rsi in self.ticker_rsi})
        #print(f"Returns: {returns}, Weights: {weights}")

        sharpe = local_portfolio(returns, weights)[-1]
        sharpe = np.array([sharpe] * len(self.tickers))          
        rew = (data_period.values[-1] - data_period.values[0]) / data_period.values[0]

        #print(f"Output Reward: {np.dot(returns, weights)}, Sharpe: {sharpe}")
        # assert correct shape and no nan
        assert returns.shape[1] == len(weights), "Length of returns and weights mismatch"
        assert not np.isnan(returns.values).any(), "Nan in returns"
        #print(f"Returns: {returns}, Weights: {weights}")
        return np.dot(returns, weights), sharpe


    '''Factor Trading Functions'''
    def update(self, stocks, curr_date):
        self.active_bars += 1
        start_date = curr_date - pd.DateOffset(days=self.lookback - 1)
        end_date = curr_date

        # Updating of Open Prices for each ticker
        for ticker in stocks:
            if len(self.open_prices[self.ticker_to_rsi[ticker]]) == 0:
                #print(f"Initial Update for {ticker} at Bar {self.active_bars}, Current length: {len(self.open_prices[self.ticker_to_rsi[ticker]])}, Current Length of Deque: {len(self.deque_returns[self.ticker_to_rsi[ticker]])}")
                self.open_prices[self.ticker_to_rsi[ticker]].append(stocks[ticker].open[0])
                self.deque_returns[self.ticker_to_rsi[ticker]].append(None)
            else:
                #print(f"Update for {ticker} at Bar {self.active_bars}, Current length: {len(self.open_prices[self.ticker_to_rsi[ticker]])}, Current Length of Deque: {len(self.deque_returns[self.ticker_to_rsi[ticker]])}")
                last_open = self.open_prices[self.ticker_to_rsi[ticker]][-1]
                self.deque_returns[self.ticker_to_rsi[ticker]].append(stocks[ticker].open[0] / last_open - 1)
                self.open_prices[self.ticker_to_rsi[ticker]].append(stocks[ticker].open[0])
        return 
    