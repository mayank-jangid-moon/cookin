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
    A trading strategy based on SMA (Simple Moving Average) crossover with RSI (Relative Strength Index) filter.
    
    This strategy generates trading signals by:
    1. Identifying bullish and bearish SMA crossovers
    2. Filtering signals using RSI overbought/oversold conditions
    3. Setting stop losses based on ATR (Average True Range)
    4. Calculating appropriate position sizes based on risk management rules
    5. Detecting and avoiding choppy market conditions
    
    The strategy is designed to:
    - Go long when the fast MA crosses above the slow MA (and RSI is not overbought)
    - Go short when the fast MA crosses below the slow MA (and RSI is not oversold)
    - Exit positions on opposing crossovers or extreme RSI values
    - Implement stop losses based on ATR
    - Avoid trading in sideways or choppy market conditions
    """
    
    def __init__(self, rsi_period=14, rsi_overbought=70, rsi_oversold=30, atr_period=14, 
                 atr_multiplier=2.0, risk_pct=1.0, adx_period=14, adx_threshold=25, 
                 ema_short=20, ema_long=50, atr_threshold_pct=0.5, bb_period=20, bb_std=2.0):
        """
        Initialize the strategy with customizable parameters.
        
        Parameters:
            rsi_period (int): Period for RSI calculation (default: 14 days)
            rsi_overbought (float): RSI threshold for overbought condition (default: 70)
            rsi_oversold (float): RSI threshold for oversold condition (default: 30)
            atr_period (int): Period for ATR calculation (default: 14 days)
            atr_multiplier (float): Multiplier applied to ATR for stop loss distance (default: 2.0)
            risk_pct (float): Risk percentage per trade of total equity (default: 1.0%)
            adx_period (int): Period for ADX calculation (default: 14 days)
            adx_threshold (float): ADX threshold for trend strength (default: 25)
            ema_short (int): Period for short EMA calculation (default: 20)
            ema_long (int): Period for long EMA calculation (default: 50)
            atr_threshold_pct (float): Minimum ATR as percentage of price for trading (default: 0.5%)
            bb_period (int): Period for Bollinger Bands calculation (default: 20)
            bb_std (float): Standard deviation for Bollinger Bands (default: 2.0)
        """
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.atr_period = atr_period  # ATR calculation period
        self.atr_multiplier = atr_multiplier  # Multiplier for ATR to set stoploss
        self.risk_pct = risk_pct  # Risk percentage per trade (1.0 means 1%)
        
        # New parameters for choppy market detection
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold  # ADX below this indicates choppy market
        self.ema_short = ema_short
        self.ema_long = ema_long
        self.atr_threshold_pct = atr_threshold_pct  # Minimum ATR % for trading
        self.bb_period = bb_period
        self.bb_std = bb_std
        
    def run(self, df: pd.DataFrame, equity: float = 10000.0) -> pd.DataFrame:
        """
        Execute the strategy on the provided price data.
        
        This is the main entry point for strategy execution. It performs two steps:
        1. Calculates all technical indicators needed for the strategy
        2. Generates trading signals based on those indicators
        
        Parameters:
            df (pd.DataFrame): Price data containing at minimum 'open', 'high', 'low', 'close' columns
            equity (float): Starting account equity for position sizing calculations (default: $10,000)
            
        Returns:
            pd.DataFrame: The input dataframe augmented with strategy indicators, signals, and position information
        """
        df = self.calculate_indicators(df)
        df = self.detect_choppy_market(df)  # Add choppy market detection
        df = self.generate_signals(df, equity)
        return df

    def calculate_indicators(self, df):
        """
        Calculate technical indicators required for the strategy.
        
        This method computes the following indicators:
        - Fast and slow Simple Moving Averages (SMA)
        - Relative Strength Index (RSI)
        - Average True Range (ATR) for stop loss calculation
        - EMAs for trend detection
        - ADX for trend strength
        - Bollinger Bands for range detection
        
        Parameters:
            df (pd.DataFrame): Price data containing at minimum 'open', 'high', 'low', 'close' columns
            
        Returns:
            pd.DataFrame: The input dataframe augmented with calculated indicators
        """
        # Calculate fast and slow moving averages
        df["fast_ma"] = ta.SMA(df["close"], timeperiod=10)
        df["slow_ma"] = ta.SMA(df["close"], timeperiod=50)
        
        # Calculate RSI
        df["rsi"] = ta.RSI(df["close"], timeperiod=self.rsi_period)
        
        # Calculate ATR for stop loss
        df["atr"] = ta.ATR(df["high"], df["low"], df["close"], timeperiod=self.atr_period)
        
        # Add new indicators for choppy market detection
        # EMAs for trend detection
        df["ema_short"] = ta.EMA(df["close"], timeperiod=self.ema_short)
        df["ema_long"] = ta.EMA(df["close"], timeperiod=self.ema_long)
        
        # ADX for trend strength
        df["adx"] = ta.ADX(df["high"], df["low"], df["close"], timeperiod=self.adx_period)
        
        # Bollinger Bands for range detection
        df["bb_upper"], df["bb_middle"], df["bb_lower"] = ta.BBANDS(
            df["close"], 
            timeperiod=self.bb_period, 
            nbdevup=self.bb_std, 
            nbdevdn=self.bb_std
        )
        
        # Calculate Bollinger Band width as percentage of price
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"] * 100
        
        # ATR as percentage of price
        df["atr_pct"] = df["atr"] / df["close"] * 100
        
        # EMA slope (rate of change) to detect flat EMAs
        df["ema_short_slope"] = df["ema_short"].pct_change(5) * 100  # 5-period slope
        df["ema_long_slope"] = df["ema_long"].pct_change(5) * 100    # 5-period slope
        
        return df
    
    def detect_choppy_market(self, df):
        """
        Detect choppy market conditions using multiple indicators.
        
        This method identifies sideways or choppy market conditions using:
        1. Flat moving averages (EMAs with minimal slope)
        2. Low ATR (below threshold)
        3. RSI oscillating around 50
        4. Low ADX (below threshold)
        5. Contracting Bollinger Bands
        
        Parameters:
            df (pd.DataFrame): Price data with calculated indicators
            
        Returns:
            pd.DataFrame: The input dataframe augmented with choppy market flags
        """
        # Initialize choppy market indicator (1 = choppy, 0 = trending)
        df["is_choppy"] = 0
        
        # Detect flat EMAs (horizontal moving averages)
        flat_emas = (abs(df["ema_short_slope"]) < 0.2) & (abs(df["ema_long_slope"]) < 0.2)
        
        # Detect low volatility using ATR
        low_volatility = df["atr_pct"] < self.atr_threshold_pct
        
        # Detect RSI around 50 (between 40-60) indicating no momentum
        neutral_rsi = (df["rsi"] > 40) & (df["rsi"] < 60)
        
        # Detect weak trend using ADX
        weak_trend = df["adx"] < self.adx_threshold
        
        # Detect tight Bollinger Bands
        tight_bands = df["bb_width"] < 5  # Typically ranges from 2% to 7% in normal markets
        
        # Combine the indicators to detect choppy market
        # A market is considered choppy when at least 3 out of 5 conditions are met
        choppy_conditions = flat_emas.astype(int) + low_volatility.astype(int) + \
                           neutral_rsi.astype(int) + weak_trend.astype(int) + \
                           tight_bands.astype(int)
        
        # Mark as choppy if at least 3 conditions are met
        df.loc[choppy_conditions >= 3, "is_choppy"] = 1
        
        # Additional detection: Price crossing EMAs frequently
        # Use a safer approach to count EMA crosses
        # Create a series to track when price crosses above/below EMA
        df["above_ema_short"] = (df["close"] > df["ema_short"]).astype(int)
        
        # Calculate when this changes (a cross occurs)
        df["ema_cross"] = df["above_ema_short"].diff().abs()
        
        # Use rolling window to count crosses in the last 10 periods
        df["ema_cross_count"] = df["ema_cross"].rolling(10).sum()
        
        # If there are many EMA crosses in a short window, it's choppy
        df.loc[df["ema_cross_count"] >= 4, "is_choppy"] = 1
        
        # Clean up intermediate columns
        df.drop(columns=["above_ema_short", "ema_cross", "ema_cross_count"], inplace=True)
        
        return df

    def generate_signals(self, df, equity=10000.0):
        """
        Generate trading signals based on calculated indicators.
        
        This method identifies entry and exit points for trades based on SMA crossovers and RSI conditions,
        while avoiding choppy market conditions.
        It also sets stop loss prices and calculates position sizes.
        
        Parameters:
            df (pd.DataFrame): Price data containing calculated indicators
            equity (float): Starting account equity for position sizing calculations (default: $10,000)
            
        Returns:
            pd.DataFrame: The input dataframe augmented with trading signals and position information
        """
        df["trade_type"] = TradeType.HOLD.value
        df["Position"] = 0
        df["entry_price"] = np.nan
        df["stoploss_price"] = np.nan
        df["position_size"] = 0.0  # Add position size column

        # Bullish crossover with RSI filter and NOT in choppy market
        bullish_cross = (
            (df["fast_ma"] > df["slow_ma"])
            & (df["fast_ma"].shift() <= df["slow_ma"].shift())
            & (df["Position"].shift().fillna(0) == 0)
            & (df["rsi"] < self.rsi_overbought)  # RSI not overbought
            & (df["is_choppy"] == 0)  # Not in choppy market
            & (df["atr_pct"] >= self.atr_threshold_pct)  # Sufficient volatility
            & (df["adx"] >= self.adx_threshold)  # Strong trend
        )

        # Bearish crossover with RSI filter and NOT in choppy market
        bearish_cross = (
            (df["fast_ma"] < df["slow_ma"])
            & (df["fast_ma"].shift() >= df["slow_ma"].shift())
            & (df["Position"].shift().fillna(0) == 0)
            & (df["rsi"] > self.rsi_oversold)  # RSI not oversold
            & (df["is_choppy"] == 0)  # Not in choppy market
            & (df["atr_pct"] >= self.atr_threshold_pct)  # Sufficient volatility
            & (df["adx"] >= self.adx_threshold)  # Strong trend
        )

        # Exit conditions: MA crossover, RSI extreme, or entering choppy market
        exit_long = (
            (df["Position"].shift().fillna(0) == 1) 
            & ((df["fast_ma"] < df["slow_ma"]) | (df["rsi"] > self.rsi_overbought) | (df["is_choppy"] == 1))
        )
        
        exit_short = (
            (df["Position"].shift().fillna(0) == -1) 
            & ((df["fast_ma"] > df["slow_ma"]) | (df["rsi"] < self.rsi_oversold) | (df["is_choppy"] == 1))
        )
        
        # Stoploss conditions - Use pd.Series.combine with OR logic explicitly
        long_stoploss = ((df["Position"].shift().fillna(0) == 1) & 
                         (df["low"] < df["stoploss_price"].shift().fillna(0)))
        short_stoploss = ((df["Position"].shift().fillna(0) == -1) & 
                         (df["high"] > df["stoploss_price"].shift().fillna(0)))
        
        # Combined exit conditions - use bitwise OR operator explicitly
        exit_long_condition = exit_long | long_stoploss
        exit_short_condition = exit_short | short_stoploss

        # Update bullish positions - Use .loc[] consistently
        if not bullish_cross.empty and bullish_cross.any():
            df.loc[bullish_cross, "Position"] = 1
            df.loc[bullish_cross, "entry_price"] = df.loc[bullish_cross, "close"]
            df.loc[bullish_cross, "stoploss_price"] = df.loc[bullish_cross, "close"] - (df.loc[bullish_cross, "atr"] * self.atr_multiplier)
            # Pass entire Series for vectorized calculation
            position_sizes = self.calculate_position_size(
                equity, 
                df.loc[bullish_cross, "close"], 
                df.loc[bullish_cross, "stoploss_price"]
            )
            df.loc[bullish_cross, "position_size"] = position_sizes
        
        # Update bearish positions - Use .loc[] consistently
        if not bearish_cross.empty and bearish_cross.any():
            df.loc[bearish_cross, "Position"] = -1
            df.loc[bearish_cross, "entry_price"] = df.loc[bearish_cross, "close"]
            df.loc[bearish_cross, "stoploss_price"] = df.loc[bearish_cross, "close"] + (df.loc[bearish_cross, "atr"] * self.atr_multiplier)
            # Pass entire Series for vectorized calculation
            position_sizes = self.calculate_position_size(
                equity, 
                df.loc[bearish_cross, "close"], 
                df.loc[bearish_cross, "stoploss_price"]
            )
            df.loc[bearish_cross, "position_size"] = position_sizes
        
        # Exit conditions - check if there are any exit conditions first
        exit_condition = exit_long_condition | exit_short_condition
        if not exit_condition.empty and exit_condition.any():
            df.loc[exit_condition, "Position"] = 0
            df.loc[exit_condition, "entry_price"] = np.nan
            df.loc[exit_condition, "stoploss_price"] = np.nan
            df.loc[exit_condition, "position_size"] = 0.0

        # Forward-fill positions and entry/stoploss prices
        df["Position"] = df["Position"].ffill().fillna(0)
        df["entry_price"] = df["entry_price"].ffill()
        df["stoploss_price"] = df["stoploss_price"].ffill()
        df["position_size"] = df["position_size"].ffill().fillna(0)

        # Calculate position changes
        position_change = df["Position"].diff().fillna(df["Position"])
        prev_position = df["Position"].shift().fillna(0)

        # Define trade types
        entry_long = position_change == 1
        # Fix: Use loc for both sides of the assignment to ensure element-wise operation
        df.loc[entry_long, "trade_type"] = [
            TradeType.REVERSE_LONG.value if prev == -1 else TradeType.LONG.value 
            for prev in prev_position[entry_long]
        ]

        entry_short = position_change == -1
        # Fix: Use loc for both sides of the assignment to ensure element-wise operation
        df.loc[entry_short, "trade_type"] = [
            TradeType.REVERSE_SHORT.value if prev == 1 else TradeType.SHORT.value 
            for prev in prev_position[entry_short]
        ]

        # The exiting positions condition might also be problematic
        exit_positions = (position_change != 0) & (df["Position"] == 0)
        if not exit_positions.empty:  # First check if there are any exit positions
            df.loc[exit_positions, "trade_type"] = TradeType.CLOSE.value

        # Clean up intermediate columns
        df.drop(columns=["Position"], inplace=True)

        return df

    def calculate_position_size(self, equity, entry_price, stoploss_price):
        """
        Calculate position size based on ATR and risk percentage
        
        Args:
            equity: Total account equity
            entry_price: Entry price of the trade
            stoploss_price: Stoploss price of the trade
            
        Returns:
            position_size: Number of units/shares to trade
        """
        # Handle the case where inputs are pandas Series
        if isinstance(entry_price, pd.Series) and isinstance(stoploss_price, pd.Series):
            # Element-wise calculations for Series
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
