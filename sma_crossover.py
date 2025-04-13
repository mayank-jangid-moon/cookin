import talib as ta
import pandas as pd
import numpy as np
from enum import Enum


class TradeType(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    REVERSE_LONG = "REVERSE_LONG"
    REVERSE_SHORT = "REVERSE_SHORT"
    CLOSE = "CLOSE"
    HOLD = "HOLD"


class Strategy:
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.calculate_indicators(df)
        df = self.generate_signals(df)
        return df

    def calculate_indicators(self, df):
        # Calculate fast and slow moving averages
        df["fast_ma"] = ta.SMA(df["close"], timeperiod=10)
        df["slow_ma"] = ta.SMA(df["close"], timeperiod=50)
        return df

    def generate_signals(self, df):
        df["trade_type"] = TradeType.HOLD.value
        df["Position"] = 0

        # Bullish crossover
        bullish_cross = (
            (df["fast_ma"] > df["slow_ma"])
            & (df["fast_ma"].shift() <= df["slow_ma"].shift())
            & (df["Position"].shift().fillna(0) == 0)
        )

        # Bearish crossover
        bearish_cross = (
            (df["fast_ma"] < df["slow_ma"])
            & (df["fast_ma"].shift() >= df["slow_ma"].shift())
            & (df["Position"].shift().fillna(0) == 0)
        )

        # Exit conditions
        exit_long = (df["Position"].shift().fillna(0) == 1) & (
            df["fast_ma"] < df["slow_ma"]
        )
        exit_short = (df["Position"].shift().fillna(0) == -1) & (
            df["fast_ma"] > df["slow_ma"]
        )

        # Update positions
        df.loc[bullish_cross, "Position"] = 1
        df.loc[bearish_cross, "Position"] = -1
        df.loc[exit_long | exit_short, "Position"] = 0

        # Forward-fill positions
        df["Position"] = df["Position"].ffill().fillna(0)

        # Calculate position changes
        position_change = df["Position"].diff().fillna(df["Position"])
        prev_position = df["Position"].shift().fillna(0)

        # Define trade types
        entry_long = position_change == 1
        df.loc[entry_long, "trade_type"] = np.where(
            prev_position[entry_long] == -1,
            TradeType.REVERSE_LONG.value,
            TradeType.LONG.value,
        )

        entry_short = position_change == -1
        df.loc[entry_short, "trade_type"] = np.where(
            prev_position[entry_short] == 1,
            TradeType.REVERSE_SHORT.value,
            TradeType.SHORT.value,
        )

        # Exiting positions
        exit_positions = (position_change != 0) & (df["Position"] == 0)
        df.loc[exit_positions, "trade_type"] = TradeType.CLOSE.value

        # Clean up intermediate columns
        df.drop(columns=["Position"], inplace=True)

        return df
