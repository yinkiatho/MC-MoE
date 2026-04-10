class DividendHistory:
    def __init__(self, ticker, data):
        self.ticker = ticker
        self.data = data
        self.agg_dividends = {} # aggregated_dividends

    def aggregate_dividends(self):
        for i in range(self.data.shape[0]):
            if (self.data.loc[i, "dividend_ex_date"] in self.agg_dividends):
                self.agg_dividends[self.data.loc[i, "dividend_ex_date"]] += self.data.loc[i, "dividend_amt"]
            else:
                self.agg_dividends[self.data.loc[i, "dividend_ex_date"]] = self.data.loc[i, "dividend_amt"]
                
        return 