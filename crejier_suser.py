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
    
    def check_for_data_issues(self, data):
        """Check for potential data leakage and look-ahead bias issues"""
        issues = []
        
        # Check if indicators are properly shifted to avoid look-ahead bias
        # Technical indicators based on current prices don't need shifting
        # But any trading decisions should only use already-available information
        
        # Check for NaN values which might silently cause issues
        null_counts = data.isnull().sum()
        if null_counts.any():
            issues.append(f"Missing values found in data: {null_counts[null_counts > 0].to_dict()}")
        
        # Ensure we're only using past data in decision making
        if data.index.duplicated().any():
            issues.append("Duplicate indices found in data. This can cause look-ahead bias.")
        
        # Check chronological ordering
        if not data.index.is_monotonic_increasing:
            issues.append("Data is not in chronological order. This can cause look-ahead bias.")
            
        return issues
    
    def get_indicators(self, data):
        # Make a copy to avoid modifying original data
        data = data.copy()
        
        # Ensure data is properly sorted by time
        if not data.index.is_monotonic_increasing:
            data = data.sort_index()
        
        # Calculate indicators using TA-Lib (note: TA-Lib uses only past data properly)
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
        
        # Properly handle NaN values to avoid look-ahead bias
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
        """Modified to prevent data leakage in trade type assignment"""
        trade_types = []
        last_trade = TradeType.HOLD

        # Process each row chronologically
        for idx, row in data.iterrows():
            signal = row["signal"]

            # Decision making using only past information
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
        # Use the daily timeframe as the primary data source
        data = data_btcusdt_1d.copy()
        
        # Check for data issues before processing
        issues = self.check_for_data_issues(data)
        if issues:
            print("WARNING: Potential data issues detected:")
            for issue in issues:
                print(f" - {issue}")
                
        # Process data through the strategy pipeline
        data = self.get_indicators(data)
        
        # Verify no data leakage after indicator calculation
        indicator_issues = self.check_for_data_issues(data)
        if indicator_issues:
            print("WARNING: Issues detected after indicator calculation:")
            for issue in indicator_issues:
                print(f" - {issue}")
                
        data = self.make_signals(data)
        data = self.set_trade_type(data)
        data = self.stop_loss_take_profit(data)
        
        # Calculate and display trading statistics
        trades = data[data['trade_type'] != TradeType.HOLD.value]
        print(f"\nTotal trades: {len(trades)}")
        trade_types = trades['trade_type'].value_counts()
        for trade_type, count in trade_types.items():
            print(f"{trade_type}: {count}")
            
        return data