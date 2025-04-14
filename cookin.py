import talib as ta
import pandas as pd
import numpy as np
from enum import Enum


class TradeType(Enum):
    """
    Enumeration defining the different types of trading actions.
    
    Attributes:
        LONG: Enter a long position
        SHORT: Enter a short position
        REVERSE_LONG: Exit a short position and enter a long position
        REVERSE_SHORT: Exit a long position and enter a short position
        CLOSE: Close any open position
        HOLD: Maintain current position (no action)
    """
    LONG = "LONG"
    SHORT = "SHORT"
    REVERSE_LONG = "REVERSE_LONG"
    REVERSE_SHORT = "REVERSE_SHORT"
    CLOSE = "CLOSE"
    HOLD = "HOLD"


class Strategy:
    """
    A high-profit trading strategy combining powerful trend filtering, dip buying, 
    and impulse control for optimal entries and exits.
    
    This strategy generates trading signals by:
    1. Using hierarchical EMAs (10,20,50) for strong trend identification
    2. Implementing smart dip buying after sharp drops followed by consolidation
    3. Using extreme RSI values for better entry and exit timing
    4. Applying HFT-inspired optimal entry/exit zones for precise trade execution
    5. Maintaining strong risk management with ATR-based stop losses
    """
    
    def __init__(self, rsi_period=10, rsi_overbought=85, rsi_oversold=15, atr_period=10, 
                 atr_multiplier=3.0, risk_pct=1.0, adx_period=18, adx_threshold=15, 
                 ema_short=10, ema_medium=20, ema_long=50, atr_threshold_pct=0.5, bb_period=20, bb_std=2.0,
                 # Dip buying parameters
                 dip_drop_pct=0.01, dip_drop_atr_mult=1.0, dip_consol_window=6, dip_consol_atr_mult=0.7,
                 # HFT parameters
                 min_trade_interval=10, mean_reversion_threshold=1.5):
        """
        Initialize the strategy with optimized parameters for high profitability.
        """
        # Core parameters
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.risk_pct = risk_pct
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        
        # EMA parameters
        self.ema_short = ema_short
        self.ema_medium = ema_medium
        self.ema_long = ema_long
        
        # Volatility parameters
        self.atr_threshold_pct = atr_threshold_pct
        self.bb_period = bb_period
        self.bb_std = bb_std
        
        # Dip buying parameters
        self.dip_drop_pct = dip_drop_pct
        self.dip_drop_atr_mult = dip_drop_atr_mult
        self.dip_consol_window = dip_consol_window
        self.dip_consol_atr_mult = dip_consol_atr_mult
        
        # HFT parameters
        self.min_trade_interval = min_trade_interval
        self.mean_reversion_threshold = mean_reversion_threshold
        
    def run(self, df: pd.DataFrame, equity: float = 10000.0) -> pd.DataFrame:
        """
        Execute the strategy on the provided price data.
        """
        df = self.calculate_indicators(df)
        df = self.detect_market_regimes(df)
        df = self.add_dip_buy_signals(df)
        df = self.add_hft_components(df)
        df = self.generate_signals(df, equity)
        return df

    def calculate_indicators(self, df):
        """
        Calculate technical indicators required for the strategy.
        """
        # Primary EMAs for trend identification (based on profitable strategy)
        df["ema_10"] = ta.EMA(df["close"], timeperiod=self.ema_short)
        df["ema_20"] = ta.EMA(df["close"], timeperiod=self.ema_medium)
        df["ema_50"] = ta.EMA(df["close"], timeperiod=self.ema_long)
        
        # Calculate RSI with optimized period
        df["rsi"] = ta.RSI(df["close"], timeperiod=self.rsi_period)
        
        # Calculate ATR for stop loss and volatility measurement
        df["atr"] = ta.ATR(df["high"], df["low"], df["close"], timeperiod=self.atr_period)
        
        # Calculate ADX for trend strength
        df["adx"] = ta.ADX(df["high"], df["low"], df["close"], timeperiod=self.adx_period)
        
        # Bollinger Bands for volatility and mean reversion
        df["bb_upper"], df["bb_middle"], df["bb_lower"] = ta.BBANDS(
            df["close"], 
            timeperiod=self.bb_period, 
            nbdevup=self.bb_std, 
            nbdevdn=self.bb_std
        )
        
        # Calculate Bollinger Band width as percentage of price
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"] * 100
        
        # Distance from price to Bollinger Bands (normalized)
        df["bb_pos"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
        
        # ATR as percentage of price for volatility measurement
        df["atr_pct"] = df["atr"] / df["close"] * 100
        
        # Price changes
        df["pct_change"] = df["close"].pct_change()
        
        # Trend alignment (hierarchical EMAs)
        df["bullish_alignment"] = (df["ema_10"] > df["ema_20"]) & (df["ema_20"] > df["ema_50"])
        df["bearish_alignment"] = (df["ema_10"] < df["ema_20"]) & (df["ema_20"] < df["ema_50"])
        
        return df
    
    def detect_market_regimes(self, df):
        """
        Detect market regimes (trending, ranging, volatile).
        """
        # Initialize market regime indicator 
        df["market_regime"] = "unknown"
        
        # Detect trending markets
        strong_trend = df["adx"] > self.adx_threshold
        bullish_trend = strong_trend & df["bullish_alignment"]
        bearish_trend = strong_trend & df["bearish_alignment"]
        
        # Detect ranging/choppy markets
        flat_market = (df["adx"] < self.adx_threshold) & (df["bb_width"] < 5)
        
        # Detect volatile markets
        volatile_market = df["atr_pct"] > (self.atr_threshold_pct * 1.5)
        
        # Apply market regime labels
        df.loc[bullish_trend, "market_regime"] = "bullish_trend"
        df.loc[bearish_trend, "market_regime"] = "bearish_trend"
        df.loc[flat_market, "market_regime"] = "ranging"
        df.loc[volatile_market & ~(bullish_trend | bearish_trend), "market_regime"] = "volatile"
        
        # Fill remaining unknowns as "undefined"
        df.loc[df["market_regime"] == "unknown", "market_regime"] = "undefined"
        
        return df
    
    def add_dip_buy_signals(self, df):
        """
        Add dip buying signals based on sharp drops followed by consolidation.
        """
        # Calculate sharp drops (like in the profitable strategy)
        sharp_pct_drop = df["pct_change"] < -self.dip_drop_pct
        sharp_atr_drop = df["pct_change"] < -(self.dip_drop_atr_mult * df["atr"] / df["close"].shift(1))
        
        # Check for consolidation after drops
        range_N = df["high"].rolling(self.dip_consol_window).max() - df["low"].rolling(self.dip_consol_window).min()
        consol = range_N < (self.dip_consol_atr_mult * df["atr"])
        
        # Identify dip buy opportunities
        df["dip_buy"] = (sharp_pct_drop.shift(1) | sharp_atr_drop.shift(1)) & consol
        
        # Identify potential reversals
        df["potential_reversal"] = ((df["close"] < df["bb_lower"]) & (df["rsi"] < 30)) | df["dip_buy"]
        
        return df
    
    def add_hft_components(self, df):
        """
        Add HFT-inspired components for optimal entry/exit points.
        """
        # Optimal value band for fair price estimation
        df["value_middle"] = df["bb_middle"]
        
        # Optimal entry/exit zones
        entry_band_width = df["atr"] * 0.8  # Slightly wider than previous
        df["optimal_buy_zone"] = df["value_middle"] - entry_band_width
        df["optimal_sell_zone"] = df["value_middle"] + entry_band_width
        
        # Mean reversion opportunities
        df["mean_reversion_buy"] = (
            (df["close"] < df["optimal_buy_zone"]).fillna(False) & 
            (df["rsi"] < 30).fillna(False)
        )
        
        df["mean_reversion_sell"] = (
            (df["close"] > df["optimal_sell_zone"]).fillna(False) & 
            (df["rsi"] > 70).fillna(False)
        )
        
        # Impulse signals for quick entries/exits
        df["impulse_up"] = (df["close"] > df["close"].shift(3)) & (df["close"] > df["ema_10"]) & (df["adx"] > self.adx_threshold)
        df["impulse_down"] = (df["close"] < df["close"].shift(3)) & (df["close"] < df["ema_10"]) & (df["adx"] > self.adx_threshold)
        
        # Time since last trade placeholder
        df["bars_since_last_trade"] = 0
        
        return df

    def generate_signals(self, df, equity=10000.0):
        """
        Generate optimized trading signals based on combined signals.
        """
        df["trade_type"] = TradeType.HOLD.value
        df["Position"] = 0
        df["entry_price"] = np.nan
        df["stoploss_price"] = np.nan
        df["position_size"] = 0.0
        df["bars_since_last_trade"] = np.nan
        
        # Track time since last trade
        last_trade_idx = -999
        
        for i in range(len(df)):
            df.loc[i, "bars_since_last_trade"] = i - last_trade_idx
        
        # === ENTRY CONDITIONS (Strong trend + optimal timing) ===
        
        # Bullish entry based on profitable strategy logic
        bullish_entry = (
            (
                # Strong trend with hierarchical EMAs
                ((df["ema_10"] > df["ema_20"]) & 
                 (df["ema_20"] > df["ema_50"]) & 
                 (df["adx"] > self.adx_threshold)) |
                # Dip buying opportunity
                df["dip_buy"] |
                # Mean reversion in strong bullish trend
                (df["mean_reversion_buy"] & df["bullish_alignment"])
            ) &
            # Common filters
            (df["Position"].shift().fillna(0) == 0) &
            (df["bars_since_last_trade"] >= self.min_trade_interval).fillna(False)
        )
        
        # Bearish entry with similar logic
        bearish_entry = (
            (
                # Strong bearish trend
                ((df["ema_10"] < df["ema_20"]) & 
                 (df["ema_20"] < df["ema_50"]) & 
                 (df["adx"] > self.adx_threshold)) |
                # Mean reversion in strong bearish trend
                (df["mean_reversion_sell"] & df["bearish_alignment"])
            ) &
            # Common filters
            (df["Position"].shift().fillna(0) == 0) &
            (df["bars_since_last_trade"] >= self.min_trade_interval).fillna(False)
        )
        
        # === EXIT CONDITIONS (Clear trend changes or profit targets) ===
        
        # Long exit conditions - EMA crossover or extreme RSI
        exit_long = (
            (df["Position"].shift().fillna(0) == 1) & 
            (
                (df["ema_20"] < df["ema_50"]).fillna(False) |  # Major trend change
                (df["rsi"] > self.rsi_overbought).fillna(False) |  # Extreme RSI
                (df["impulse_down"] & (df["close"] < df["ema_10"])).fillna(False)  # Impulse down
            )
        )
        
        # Short exit conditions
        exit_short = (
            (df["Position"].shift().fillna(0) == -1) & 
            (
                (df["ema_20"] > df["ema_50"]).fillna(False) |  # Major trend change
                (df["rsi"] < self.rsi_oversold).fillna(False) |  # Extreme RSI
                (df["impulse_up"] & (df["close"] > df["ema_10"])).fillna(False)  # Impulse up
            )
        )
        
        # Stoploss conditions
        long_stoploss = (
            (df["Position"].shift().fillna(0) == 1) & 
            (df["low"] < df["stoploss_price"].shift().fillna(0)).fillna(False)
        )
        
        short_stoploss = (
            (df["Position"].shift().fillna(0) == -1) & 
            (df["high"] > df["stoploss_price"].shift().fillna(0)).fillna(False)
        )
        
        # Combined exit conditions
        exit_long_condition = exit_long | long_stoploss
        exit_short_condition = exit_short | short_stoploss
        
        # === APPLY ENTRY SIGNALS ===
        
        # Update bullish positions
        if not bullish_entry.empty and bullish_entry.any():
            df.loc[bullish_entry, "Position"] = 1
            df.loc[bullish_entry, "entry_price"] = df.loc[bullish_entry, "close"]
            
            # Set stop loss based on ATR
            df.loc[bullish_entry, "stoploss_price"] = (
                df.loc[bullish_entry, "entry_price"] - 
                df.loc[bullish_entry, "atr"] * self.atr_multiplier
            )
            
            # Calculate position sizes
            position_sizes = self.calculate_position_size(
                equity, 
                df.loc[bullish_entry, "entry_price"], 
                df.loc[bullish_entry, "stoploss_price"]
            )
            df.loc[bullish_entry, "position_size"] = position_sizes
            
            # Update last trade index
            for idx in bullish_entry[bullish_entry].index:
                last_trade_idx = idx
                indices_to_update = df.index[df.index > idx]
                if len(indices_to_update) > 0:
                    df.loc[indices_to_update, "bars_since_last_trade"] = indices_to_update - idx
        
        # Update bearish positions with similar logic
        if not bearish_entry.empty and bearish_entry.any():
            df.loc[bearish_entry, "Position"] = -1
            df.loc[bearish_entry, "entry_price"] = df.loc[bearish_entry, "close"]
            
            df.loc[bearish_entry, "stoploss_price"] = (
                df.loc[bearish_entry, "entry_price"] + 
                df.loc[bearish_entry, "atr"] * self.atr_multiplier
            )
            
            position_sizes = self.calculate_position_size(
                equity, 
                df.loc[bearish_entry, "entry_price"], 
                df.loc[bearish_entry, "stoploss_price"]
            )
            df.loc[bearish_entry, "position_size"] = position_sizes
            
            for idx in bearish_entry[bearish_entry].index:
                last_trade_idx = idx
                indices_to_update = df.index[df.index > idx]
                if len(indices_to_update) > 0:
                    df.loc[indices_to_update, "bars_since_last_trade"] = indices_to_update - idx
        
        # Apply exit conditions
        exit_condition = exit_long_condition | exit_short_condition
        if not exit_condition.empty and exit_condition.any():
            df.loc[exit_condition, "Position"] = 0
            df.loc[exit_condition, "entry_price"] = np.nan
            df.loc[exit_condition, "stoploss_price"] = np.nan
            df.loc[exit_condition, "position_size"] = 0.0
            
            for idx in exit_condition[exit_condition].index:
                last_trade_idx = idx
                indices_to_update = df.index[df.index > idx]
                if len(indices_to_update) > 0:
                    df.loc[indices_to_update, "bars_since_last_trade"] = indices_to_update - idx
        
        # Forward-fill positions and related data
        df["Position"] = df["Position"].ffill().fillna(0)
        df["entry_price"] = df["entry_price"].ffill()
        df["stoploss_price"] = df["stoploss_price"].ffill()
        df["position_size"] = df["position_size"].ffill().fillna(0)
        
        # Calculate position changes for trade type determination
        position_change = df["Position"].diff().fillna(df["Position"])
        prev_position = df["Position"].shift().fillna(0)
        
        # Define trade types
        entry_long = position_change == 1
        df.loc[entry_long, "trade_type"] = [
            TradeType.REVERSE_LONG.value if prev == -1 else TradeType.LONG.value 
            for prev in prev_position[entry_long]
        ]
        
        entry_short = position_change == -1
        df.loc[entry_short, "trade_type"] = [
            TradeType.REVERSE_SHORT.value if prev == 1 else TradeType.SHORT.value 
            for prev in prev_position[entry_short]
        ]
        
        # Handle exit signals
        exit_positions = (position_change != 0) & (df["Position"] == 0)
        if not exit_positions.empty:
            df.loc[exit_positions, "trade_type"] = TradeType.CLOSE.value
        
        # Clean up intermediate columns
        df.drop(columns=["Position"], inplace=True)
        
        return df

    def calculate_position_size(self, equity, entry_price, stoploss_price):
        """
        Calculate position size based on risk percentage.
        """
        # Handle the case where inputs are pandas Series
        if isinstance(entry_price, pd.Series) and isinstance(stoploss_price, pd.Series):
            risk_amount = equity * (self.risk_pct / 100)
            risk_per_unit = (entry_price - stoploss_price).abs()
            
            # Calculate position size only where risk_per_unit is valid
            position_size = pd.Series(0.0, index=risk_per_unit.index)
            valid_mask = (risk_per_unit > 0) & ~risk_per_unit.isna()
            position_size.loc[valid_mask] = risk_amount / risk_per_unit.loc[valid_mask]
            
            return position_size.round(2)
        else:
            # Scalar calculation
            risk_amount = equity * (self.risk_pct / 100)
            risk_per_unit = abs(entry_price - stoploss_price)
            
            # Avoid division by zero or NaN
            if risk_per_unit == 0 or pd.isna(risk_per_unit):
                return 0
                
            position_size = risk_amount / risk_per_unit
            return np.round(position_size, 2)
