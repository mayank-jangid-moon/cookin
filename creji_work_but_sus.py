import pandas as pd
import numpy as np
import talib
from enum import Enum

class TradeType(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    REVERSE_LONG = "REVERSE_LONG"
    REVERSE_SHORT = "REVERSE_SHORT"
    CLOSE = "CLOSE"
    HOLD = "HOLD"

class Strategy:
    def __init__(self, atr_multiplier=3, rsi_overbought=70, rsi_oversold=30, adx_threshold=20):
        self.atr_multiplier = atr_multiplier
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.adx_threshold = adx_threshold
    
    def get_indicators(self, data):
        data["ATR"] = talib.ATR(data["high"], data["low"], data["close"], timeperiod=18).fillna(0)
        data["EMA_20"] = talib.EMA(data["close"], timeperiod=10)
        data["EMA_50"] = talib.EMA(data["close"], timeperiod=25)
        data["EMA_200"] = talib.EMA(data["close"], timeperiod=50)
        data["RSI"] = talib.RSI(data["close"], timeperiod=25)
        data["ADX"] = talib.ADX(data["high"], data["low"], data["close"], timeperiod=18)

        macd, macd_signal, _ = talib.MACD(data["close"], fastperiod=12, slowperiod=26, signalperiod=9)
        data["MACD"] = macd
        data["MACD_signal"] = macd_signal

        data["BB_upper"], data["BB_middle"], data["BB_lower"] = talib.BBANDS(
            data["close"], timeperiod=5, nbdevup=1, nbdevdn=1, matype=0
        )
        return data.dropna()
    
    def make_signals(self, data):
        data["signal"] = 0
        
        # Fixed bullish conditions - made less restrictive
        bullish_1 = (data["EMA_20"] > data["EMA_50"]) & (data["ADX"] > self.adx_threshold)
        bullish_2 = (data["MACD"] > data["MACD_signal"]) | (data["RSI"] > self.rsi_oversold)
        
        # Fixed bearish conditions - corrected redundant check and made less restrictive
        bearish_1 = (data["EMA_20"] < data["EMA_50"]) & (data["ADX"] > self.adx_threshold)
        bearish_2 = (data["MACD"] < data["MACD_signal"]) | (data["RSI"] < self.rsi_overbought)

        # Assign signals
        data.loc[bullish_1 & bullish_2, "signal"] = 1
        data.loc[bearish_1 & bearish_2, "signal"] = -1
        
        return data

    def set_trade_type(self, data):
        trade_types = []
        last_trade = TradeType.HOLD

        for _, row in data.iterrows():
            signal = row["signal"]

            if last_trade in [TradeType.HOLD, TradeType.CLOSE]:
                if signal == 1:
                    trade_types.append(TradeType.LONG.value)
                    last_trade = TradeType.LONG
                elif signal == -1:
                    trade_types.append(TradeType.SHORT.value)
                    last_trade = TradeType.SHORT
                else:
                    trade_types.append(TradeType.HOLD.value)
            
            elif last_trade == TradeType.SHORT:
                if signal == 1:
                    trade_types.append(TradeType.REVERSE_LONG.value)
                    last_trade = TradeType.LONG  # Changed to LONG from REVERSE_LONG
                elif signal == 0:
                    trade_types.append(TradeType.HOLD.value)
                else:
                    # Keep the SHORT position while signal is still -1
                    trade_types.append(TradeType.HOLD.value)

            elif last_trade == TradeType.LONG:
                if signal == -1:
                    trade_types.append(TradeType.REVERSE_SHORT.value)
                    last_trade = TradeType.SHORT  # Changed to SHORT from REVERSE_SHORT
                elif signal == 0:
                    trade_types.append(TradeType.HOLD.value)
                else:
                    # Keep the LONG position while signal is still 1
                    trade_types.append(TradeType.HOLD.value)
                    
            # Handle REVERSE_LONG and REVERSE_SHORT properly
            elif last_trade in [TradeType.REVERSE_LONG, TradeType.REVERSE_SHORT]:
                if signal == 0:
                    trade_types.append(TradeType.HOLD.value)
                    # Maintain the current direction
                    last_trade = TradeType.LONG if last_trade == TradeType.REVERSE_LONG else TradeType.SHORT
                else:
                    trade_types.append(TradeType.HOLD.value)
            else:
                trade_types.append(TradeType.HOLD.value)

        data["trade_type"] = trade_types
        return data

    def stop_loss_take_profit(self, data):
        sl_values, tp_values = [], []
        for _, row in data.iterrows():
            atr = row["ATR"]
            close_price = row["close"]

            if row["trade_type"] in [TradeType.LONG.value, TradeType.REVERSE_LONG.value]:
                sl_values.append(close_price - self.atr_multiplier*atr)
                tp_values.append(close_price + 2*self.atr_multiplier*atr)
            
            elif row["trade_type"] in [TradeType.SHORT.value, TradeType.REVERSE_SHORT.value]:
                sl_values.append(close_price + self.atr_multiplier*atr)
                tp_values.append(close_price - 2*self.atr_multiplier*atr)
            
            else:
                sl_values.append(None)
                tp_values.append(None)
        
        data["stop_loss"] = sl_values
        data["take_profit"] = tp_values
        return data
        
    def run(self, data_btcusdt_1d: pd.DataFrame) -> pd.DataFrame:
        # Use the 4h timeframe as the primary data source
        data = data_btcusdt_1d.copy()
        data = self.get_indicators(data)
        data = self.make_signals(data)
        data = self.set_trade_type(data)
        data = self.stop_loss_take_profit(data)
        return data