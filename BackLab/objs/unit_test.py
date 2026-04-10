from stock import Stock
from series import Series
from stock import StockInitializer

class UnitTest():
    def test_stock(self):
        stock_bulk = {}
        ticker1 = "SPY"
        index1 = 0
        stock_bulk[index1] = Stock(ticker1, index1)
        print(stock_bulk)

        tickers = ["SPY", "QQQ"]
        z = StockInitializer.run(tickers)
        print(z)
        
    def test_series(self):
        s = Series([5,4,3,2,1])
        print(s[0])
        print(s[1])
        print(s[2])
        print(s[3])
        print(s[4])
        s.add(100)
        print(s)


if __name__ == "__main__":
    UT = UnitTest()
    UT.test_stock()
    UT.test_series()

