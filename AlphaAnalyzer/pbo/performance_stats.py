import numpy as np
import pandas as pd
from scipy.stats import rankdata

class PerformanceStatistics:
    @staticmethod
    def run(matrix, method = "sharpe", risk_free = 0.0, one_year_bars=252):
        if (method == "sharpe"):
            return PerformanceStatistics.sharpe(matrix, risk_free, one_year_bars)

    @staticmethod
    def sharpe(matrix, risk_free, one_year_bars):
        # matrix is a 2d-list where each column denotes the same strategy with different parameters; each row denotes daily return
        np_matrix = np.array(matrix)
        daily_vols = np_matrix.std(axis = 0, ddof = 1)
        yearly_vols = daily_vols*pow(one_year_bars,0.5)
        nav = np_matrix + 1 # nav
        cum_prod_nav = np.cumprod(nav, axis = 0)
        final_nav = cum_prod_nav[-1]
        cagr = np.power(final_nav, one_year_bars/len(matrix)) - 1
        sharpe = (cagr-risk_free)/yearly_vols

        return sharpe

    @staticmethod
    def rank_ps(ps_matrix):
        ranked_ps = rankdata(ps_matrix, method='min')
        return ranked_ps