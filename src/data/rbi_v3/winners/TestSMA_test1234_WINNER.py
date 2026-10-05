
import numpy as np
import talib
from backtesting import Strategy

class TestSMA(Strategy):
    fast_p = 10
    slow_p = 30
    def init(self):
        self.fast = self.I(talib.SMA, self.data.Close, self.fast_p)
        self.slow = self.I(talib.SMA, self.data.Close, self.slow_p)
    def next(self):
        if self.fast[-2] < self.slow[-2] and self.fast[-1] > self.slow[-1]:
            self.buy()
