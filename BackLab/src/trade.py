from collections import deque
import pandas as pd
import time 

class Order:
    def __init__(self, ticker, date_init, quantity, price):
        self.ticker = ticker
        self.date_init = date_init
        self.quantity = quantity
        self.price = price

    def __str__(self):
        return f"Ticker: {self.ticker}, Date Init: {self.date_init}, Quantity: {self.quantity}, Price: {self.price}."
    
class Trade:
    def __init__(self, ticker, long_date, short_date, quantity_traded, price_long, price_short):
        self.ticker = ticker
        self.long_date = long_date
        self.short_date = short_date
        self.quantity_traded = quantity_traded
        self.price_long = price_long
        self.price_short = price_short
    
    def __str__(self):
        return f"TRADE DONE: Ticker: {self.ticker}, LongDate: {self.long_date}, ShortDate: {self.short_date}, "\
             f"QuantityTraded: {self.quantity_traded}, PriceLong: {self.price_long}, PriceShort: {self.price_short}, "\
             f"PnL: {(self.price_short-self.price_long)*self.quantity_traded}."

class TradeRecords:
    def __init__(self):
        self.book = []

    def print(self):
        for i in range(len(self.book)):
            print(self.book[i])

        return 
    
    def append(self, trade: Trade):
        self.book.append(trade)
        return 

class TotalOrdersBook:
    def __init__(self, tickers):
        self.orders_book = {}
        self.tickers = tickers
        for ticker in tickers:
            self.orders_book[ticker] = OrdersBook()

    def __getitem__(self, ticker):
        return self.orders_book[ticker]

    def print_ob(self):
        for ticker in self.tickers:
            self.orders_book[ticker].print()
        return 
    
    def print_trades(self):
        for ticker in self.tickers:
            self.orders_book[ticker].trade_records.print()
    
class OrdersBook:
    def __init__(self):
        self.long_orders_q = deque()
        self.short_orders_q = deque()
        self.trade_records = TradeRecords()

    def add_order(self, order: Order):
        if (order.quantity == 0):
            raise Exception("Order quantity shall not zero.")
        
        if (order.quantity > 0):
            self.long_orders_q.append(order)
        else:
            self.short_orders_q.append(order)

        self.match_orders()
        return 
    
    def match_orders(self):
        while (len(self.long_orders_q) > 0 and len(self.short_orders_q) > 0):
            matching_trade_units = min(self.long_orders_q[0].quantity, abs(self.short_orders_q[0].quantity))
            self.trade_records.append(Trade(self.long_orders_q[0].ticker, self.long_orders_q[0].date_init, self.short_orders_q[0].date_init,
                                            matching_trade_units, self.long_orders_q[0].price, self.short_orders_q[0].price))
            
            self.long_orders_q[0].quantity -= matching_trade_units
            self.short_orders_q[0].quantity += matching_trade_units

            if (self.long_orders_q[0].quantity == 0):
                self.long_orders_q.popleft()

            if (self.short_orders_q[0].quantity == 0):
                self.short_orders_q.popleft()

        return 
    
    def print(self):
        if (len(self.long_orders_q) != 0):
            print("--- LONG ORDER BOOK ---")
            for i in range(len(self.long_orders_q)):
                print(self.long_orders_q[i])

        if (len(self.short_orders_q) != 0):
            print("--- SHORT ORDER BOOK ---")
            for i in range(len(self.short_orders_q)):
                print(self.short_orders_q[i])

        print("--- END OF PRINTING ORDER BOOK ---")

if __name__ == "__main__":
    start = time.time()
    order_book = OrdersBook()

    o = Order(ticker = "SPY", date_init = "2020-01-01", quantity = 100, price = 505)
    order_book.add_order(o)
    o = Order(ticker = "SPY", date_init = "2020-01-01", quantity = 200, price = 510)
    order_book.add_order(o)
    o = Order(ticker = "SPY", date_init = "2021-01-01", quantity = -50, price = 520)
    order_book.add_order(o)
    order_book.print()
    order_book.trade_records.print()
    o = Order(ticker = "SPY", date_init = "2021-01-01", quantity = -300, price = 520)
    order_book.add_order(o)
    order_book.print()
    order_book.trade_records.print()


    end = time.time()
    print(f"\n\nelapsed time: {round(end-start,5)} seconds")

