# region imports
from AlgorithmImports import *
import numpy as np
from collections import deque
from datetime import datetime, timedelta
from linear_regression import linear_regression
# endregion

TICKER_A = "V"
TICKER_B = "MA"
YEAR = 2024

class CsvSecondBar(PythonData):
    def get_source(self, config, date, is_live_mode):
        # Use the ticker name passed in add_data to choose the file
        file_map = {
            "VCSV": f"/Lean/Data/custom/{TICKER_A}_1s_bars_{YEAR}.csv",
            "MACSV": f"/Lean/Data/custom/{TICKER_B}_1s_bars_{YEAR}.csv"
        }

        return SubscriptionDataSource(
            file_map[config.symbol.value],
            SubscriptionTransportMedium.LOCAL_FILE
        )

    def reader(self, config, line, date, is_live_mode):
        if not line or line.startswith("timestamp"):
            return None

        parts = line.split(",")
        if len(parts) < 6:
            return None

        bar = CsvSecondBar()
        bar.symbol = config.symbol

        ts = datetime.fromisoformat(parts[0])

        # LEAN expects datetimes on the custom object
        bar.time = ts.replace(tzinfo=None)
        bar.end_time = bar.time + timedelta(seconds=1)

        o = float(parts[1])
        h = float(parts[2])
        l = float(parts[3])
        c = float(parts[4])
        v = float(parts[5])

        bar.value = c
        bar["Open"] = o
        bar["High"] = h
        bar["Low"] = l
        bar["Close"] = c
        bar["Volume"] = v

        return bar


class MyProject(QCAlgorithm):

    def _init_security(self, security: Security) -> None:
        security.set_fee_model(ConstantFeeModel(0.5, "USD"))

    def initialize(self):
        self.add_security_initializer(self._init_security)
        self.set_start_date(YEAR, 1, 2)
        self.set_end_date(YEAR, 12, 31)
        self.set_cash(100000)

        # Custom CSV subscriptions
        self.v_symbol = self.add_data(CsvSecondBar, "VCSV", Resolution.SECOND).symbol
        self.ma_symbol = self.add_data(CsvSecondBar, "MACSV", Resolution.SECOND).symbol

        self.buffer = deque(maxlen=3600)
        self.rolling_a = deque(maxlen=300)
        self.rolling_b = deque(maxlen=300)
        self.prev_v_close = None
        self.prev_ma_close = None
        self.in_trade = False
        self.trade_direction = None

        alpha, beta, P, Q, R = linear_regression(TICKER_A, TICKER_B, YEAR)
        self.state = np.array([alpha, beta])
        self.P = P
        self.Q = Q
        self.R = R


    def on_data(self, data: Slice):
        if not data.contains_key(self.v_symbol) or not data.contains_key(self.ma_symbol):
            return
        
        v_bar = data[self.v_symbol]
        ma_bar = data[self.ma_symbol]

        v_close = float(v_bar.Close)
        ma_close = float(ma_bar.Close)

        log_ma = np.log(ma_close)
        log_v = np.log(v_close)


        H = np.array([1.0, log_ma])


        self.P = self.P + self.Q


        error = log_v - (self.state[0] + self.state[1] * log_ma)


        S = H @ self.P @ H + self.R


        K = self.P @ H / S


        self.state = self.state + K * error

        self.P = self.P - np.outer(K, H) @ self.P


        spread = log_v - (self.state[0] + self.state[1] * log_ma)
        self.buffer.append(spread)

        if self.prev_v_close is not None and self.prev_ma_close is not None:
            r_v = np.log(v_close / self.prev_v_close)
            r_ma = np.log(ma_close / self.prev_ma_close)

            self.rolling_a.append(r_v)
            self.rolling_b.append(r_ma)

        self.prev_v_close = v_close
        self.prev_ma_close = ma_close

        if len(self.rolling_a) < 300 or len(self.rolling_b) < 300:
            return

        if len(self.buffer) < 1800:
            return
        
        # Calculate pearson correlation
        rolling_arr_a = np.array(self.rolling_a)
        rolling_arr_b = np.array(self.rolling_b)

        corr_matrix = np.corrcoef(rolling_arr_a, rolling_arr_b)
        corr = corr_matrix[0, 1]

        if np.isnan(corr):
            return
        allow_entry = corr >= 0.5

        #Calculate z score
        arr = np.array(self.buffer)
        std = arr.std()
        if std == 0:
            return
        z_score = (arr[-1] - arr.mean()) / std

        # Entry logic
        if z_score > 2.0 and not self.in_trade and allow_entry:
            self.in_trade = True
            self.trade_direction = "SELL"
            self.set_holdings(self.v_symbol, -0.35)
            self.set_holdings(self.ma_symbol, 0.35)

        elif z_score < -2.0 and not self.in_trade and allow_entry:
            self.in_trade = True
            self.trade_direction = "BUY"
            self.set_holdings(self.v_symbol, 0.35)
            self.set_holdings(self.ma_symbol, -0.35)

        elif self.in_trade:
            # Stop loss
            if ((self.trade_direction == "SELL" and z_score > 5.0) or
                (self.trade_direction == "BUY" and z_score < -5.0)):
                self.in_trade = False
                self.liquidate()

            # Take profit
            elif abs(z_score) < 0.5:
                self.in_trade = False
                self.liquidate()