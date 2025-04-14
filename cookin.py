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
    An advanced multi-timeframe trading strategy with ultra-low drawdown characteristics.
    
    Key features:
    1. Triple timeframe analysis (fast, medium, slow) for precision entries
    2. Adaptive position sizing based on volatility and drawdown control
    3. Partial profit-taking to lock in gains and reduce drawdown
    4. Trailing stops and volatility-adjusted exits for faster recovery
    5. Market regime detection with regime-specific trading rules
    6. Aggressive profit-booking during uncertain regimes
    7. Multiple entry types (trend, dip, breakout, momentum) for increased trade count
    8. Global risk budget management to prevent excessive drawdowns
    """
    
    def __init__(self, rsi_period=8, rsi_overbought=80, rsi_oversold=20, atr_period=10, 
                 atr_multiplier=2.0, risk_pct=0.5, adx_period=14, adx_threshold=15, 
                 ema_fast=8, ema_medium=20, ema_slow=50, 
                 ema_ultrafast=3, atr_threshold_pct=0.5, bb_period=20, bb_std=2.0,
                 # Dip buying parameters
                 dip_drop_pct=0.01, dip_drop_atr_mult=0.8, dip_consol_window=5, dip_consol_atr_mult=0.7,
                 # Advanced risk management
                 max_pos_volatility_ratio=1.5, profit_target_r_multiple=1.5, 
                 partial_exit_r_multiple=0.8, min_trade_interval=5, 
                 # Drawdown control
                 max_risk_per_trade_pct=0.5, max_correlated_trades=3, 
                 drawdown_scaling_factor=0.5, max_open_risk_pct=2.0,
                 # Advanced exits
                 trailing_stop_atr_multiple=1.0, time_stop_bars=8,
                 # Breakout parameters
                 breakout_lookback=20, breakout_atr_mult=1.2,
                 # Recovery parameters
                 recovery_mode_drawdown_pct=10.0, recovery_risk_factor=0.5):
        """
        Initialize the strategy with optimized parameters for superior performance.
        """
        # Core parameters
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.risk_pct = risk_pct  # Reduced base risk per trade
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        
        # EMA parameters - multi-timeframe approach
        self.ema_fast = ema_fast
        self.ema_medium = ema_medium
        self.ema_slow = ema_slow
        self.ema_ultrafast = ema_ultrafast
        
        # Volatility parameters
        self.atr_threshold_pct = atr_threshold_pct
        self.bb_period = bb_period
        self.bb_std = bb_std
        
        # Dip buying parameters - optimized for more opportunities
        self.dip_drop_pct = dip_drop_pct
        self.dip_drop_atr_mult = dip_drop_atr_mult
        self.dip_consol_window = dip_consol_window
        self.dip_consol_atr_mult = dip_consol_atr_mult
        
        # Advanced risk management - critical for drawdown control
        self.max_pos_volatility_ratio = max_pos_volatility_ratio
        self.profit_target_r_multiple = profit_target_r_multiple
        self.partial_exit_r_multiple = partial_exit_r_multiple
        self.min_trade_interval = min_trade_interval
        
        # Drawdown control - essential for meeting our targets
        self.max_risk_per_trade_pct = max_risk_per_trade_pct
        self.max_correlated_trades = max_correlated_trades
        self.drawdown_scaling_factor = drawdown_scaling_factor
        self.max_open_risk_pct = max_open_risk_pct
        
        # Advanced exits - faster recovery
        self.trailing_stop_atr_multiple = trailing_stop_atr_multiple
        self.time_stop_bars = time_stop_bars
        
        # Breakout parameters - more trade opportunities
        self.breakout_lookback = breakout_lookback
        self.breakout_atr_mult = breakout_atr_mult
        
        # Recovery parameters
        self.recovery_mode_drawdown_pct = recovery_mode_drawdown_pct
        self.recovery_risk_factor = recovery_risk_factor
        
        # Runtime metrics - these will be updated during execution
        self.current_drawdown = 0.0
        self.max_reached_equity = 0.0
        self.in_recovery_mode = False
        self.trade_history = []
        self.current_open_risk = 0.0
        
    def run(self, df: pd.DataFrame, equity: float = 10000.0) -> pd.DataFrame:
        """
        Execute the strategy on the provided price data.
        """
        # Initialize runtime metrics
        self.max_reached_equity = equity
        self.current_drawdown = 0.0
        self.in_recovery_mode = False
        self.trade_history = []
        self.current_open_risk = 0.0
        
        # Calculate all required indicators
        df = self.calculate_indicators(df)
        df = self.detect_market_regimes(df)
        df = self.add_breakout_signals(df)
        df = self.add_dip_buy_signals(df)
        df = self.add_hft_components(df)
        df = self.identify_volatility_regimes(df)
        df = self.generate_signals(df, equity)
        
        return df

    def calculate_indicators(self, df):
        """
        Calculate technical indicators for multi-timeframe analysis.
        """
        # Multiple timeframe EMAs for precision entry/exit
        df["ema3"] = ta.EMA(df["close"], timeperiod=self.ema_ultrafast)
        df["ema8"] = ta.EMA(df["close"], timeperiod=self.ema_fast)
        df["ema20"] = ta.EMA(df["close"], timeperiod=self.ema_medium)
        df["ema50"] = ta.EMA(df["close"], timeperiod=self.ema_slow)
        
        # Short-term momentum indicators
        df["rsi"] = ta.RSI(df["close"], timeperiod=self.rsi_period)
        df["rsi_fast"] = ta.RSI(df["close"], timeperiod=4) # Ultra-fast RSI for quick signals
        
        # Volatility indicators
        df["atr"] = ta.ATR(df["high"], df["low"], df["close"], timeperiod=self.atr_period)
        df["atr_pct"] = df["atr"] / df["close"] * 100
        
        # Trend strength
        df["adx"] = ta.ADX(df["high"], df["low"], df["close"], timeperiod=self.adx_period)
        df["di_plus"] = ta.PLUS_DI(df["high"], df["low"], df["close"], timeperiod=self.adx_period)
        df["di_minus"] = ta.MINUS_DI(df["high"], df["low"], df["close"], timeperiod=self.adx_period)
        
        # Bollinger Bands for volatility and mean reversion
        df["bb_upper"], df["bb_middle"], df["bb_lower"] = ta.BBANDS(
            df["close"], timeperiod=self.bb_period, nbdevup=self.bb_std, nbdevdn=self.bb_std
        )
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"] * 100
        df["bb_pos"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
        
        # Price changes for momentum analysis
        df["pct_change"] = df["close"].pct_change()
        df["pct_change_3"] = df["close"].pct_change(3)
        df["pct_change_5"] = df["close"].pct_change(5)
        
        # Multi-timeframe trend alignment
        df["trend_aligned_bull"] = (df["ema3"] > df["ema8"]) & (df["ema8"] > df["ema20"]) & (df["ema20"] > df["ema50"])
        df["trend_aligned_bear"] = (df["ema3"] < df["ema8"]) & (df["ema8"] < df["ema20"]) & (df["ema20"] < df["ema50"])
        
        # Fast-medium alignment (more trades)
        df["fast_aligned_bull"] = (df["ema3"] > df["ema8"]) & (df["ema8"] > df["ema20"])
        df["fast_aligned_bear"] = (df["ema3"] < df["ema8"]) & (df["ema8"] < df["ema20"])
        
        # MACD for momentum confirmation
        df["macd"], df["macd_signal"], df["macd_hist"] = ta.MACD(
            df["close"], fastperiod=12, slowperiod=26, signalperiod=9
        )
        
        # Momentum indicators
        df["momentum"] = df["close"] - df["close"].shift(4)
        df["momentum_pct"] = df["momentum"] / df["close"].shift(4) * 100
        
        # Keltner Channels for breakout detection
        df["keltner_middle"] = ta.EMA(df["close"], timeperiod=20)
        df["keltner_upper"] = df["keltner_middle"] + df["atr"] * 2
        df["keltner_lower"] = df["keltner_middle"] - df["atr"] * 2
        
        return df
    
    def detect_market_regimes(self, df):
        """
        Detect market regimes with improved precision for regime-specific rules.
        """
        # Initialize market regime indicator 
        df["market_regime"] = "unknown"
        
        # Strong trend detection
        strong_bull = (df["adx"] > 25) & df["trend_aligned_bull"] & (df["di_plus"] > df["di_minus"])
        strong_bear = (df["adx"] > 25) & df["trend_aligned_bear"] & (df["di_minus"] > df["di_plus"])
        
        # Weak trend detection (still tradable but with caution)
        weak_bull = (df["adx"] > self.adx_threshold) & df["fast_aligned_bull"] & (df["di_plus"] > df["di_minus"])
        weak_bear = (df["adx"] > self.adx_threshold) & df["fast_aligned_bear"] & (df["di_minus"] > df["di_plus"])
        
        # Ranging market detection (requires different approach)
        ranging = (df["adx"] < self.adx_threshold) & (df["bb_width"] < 4.5)
        
        # Volatile/uncertain market detection (reduce exposure)
        volatile = (df["atr_pct"] > 1.5 * df["atr_pct"].rolling(20).mean()) | (df["bb_width"] > 2 * df["bb_width"].rolling(20).mean())
        
        # Apply market regime labels
        df.loc[strong_bull, "market_regime"] = "strong_bull"
        df.loc[strong_bear, "market_regime"] = "strong_bear"
        df.loc[weak_bull & ~strong_bull, "market_regime"] = "weak_bull"
        df.loc[weak_bear & ~strong_bear, "market_regime"] = "weak_bear"
        df.loc[ranging & ~(weak_bull | weak_bear | strong_bull | strong_bear), "market_regime"] = "ranging"
        df.loc[volatile & ~(strong_bull | strong_bear), "market_regime"] = "volatile"
        
        # Fill remaining unknowns as "uncertain" - trade with extreme caution
        df.loc[df["market_regime"] == "unknown", "market_regime"] = "uncertain"
        
        return df
    
    def identify_volatility_regimes(self, df):
        """
        Identify volatility regimes for position sizing and risk management.
        """
        # Calculate 20-day rolling volatility
        df["vol_20d"] = df["atr_pct"].rolling(20).mean()
        
        # Normalize current volatility relative to recent history
        df["vol_ratio"] = df["atr_pct"] / df["vol_20d"]
        
        # Define volatility regimes
        df["vol_regime"] = "normal"
        df.loc[df["vol_ratio"] > 1.5, "vol_regime"] = "high"
        df.loc[df["vol_ratio"] < 0.7, "vol_regime"] = "low"
        
        # Calculate volatility-adjusted position size multiplier
        df["vol_pos_multiplier"] = 1.0
        df.loc[df["vol_regime"] == "high", "vol_pos_multiplier"] = 0.7
        df.loc[df["vol_regime"] == "low", "vol_pos_multiplier"] = 1.3
        
        return df
    
    def add_breakout_signals(self, df):
        """
        Add breakout signals for increased trade opportunities.
        """
        # Calculate rolling highs and lows
        df["n_day_high"] = df["high"].rolling(self.breakout_lookback).max()
        df["n_day_low"] = df["low"].rolling(self.breakout_lookback).min()
        
        # Calculate proximity to highs/lows
        df["high_proximity"] = 1 - ((df["n_day_high"] - df["close"]) / df["n_day_high"])
        df["low_proximity"] = 1 - ((df["close"] - df["n_day_low"]) / df["close"])
        
        # Detect consolidation before breakout
        df["range_20d"] = (df["high"].rolling(20).max() - df["low"].rolling(20).min()) / df["close"]
        df["range_5d"] = (df["high"].rolling(5).max() - df["low"].rolling(5).min()) / df["close"]
        df["is_consolidated"] = df["range_5d"] < 0.5 * df["range_20d"]
        
        # Identify breakout signals
        df["breakout_up"] = (
            (df["close"] > df["n_day_high"].shift(1)) & 
            (df["close"] > df["keltner_upper"]) & 
            (df["volume"] > df["volume"].rolling(20).mean() * 1.2) &
            (df["adx"] > 20)
        )
        
        df["breakout_down"] = (
            (df["close"] < df["n_day_low"].shift(1)) & 
            (df["close"] < df["keltner_lower"]) & 
            (df["volume"] > df["volume"].rolling(20).mean() * 1.2) &
            (df["adx"] > 20)
        )
        
        # Identify high-probability breakouts (consolidated + volume surge)
        df["high_prob_breakout_up"] = df["breakout_up"] & df["is_consolidated"]
        df["high_prob_breakout_down"] = df["breakout_down"] & df["is_consolidated"]
        
        return df
    
    def add_dip_buy_signals(self, df):
        """
        Add enhanced dip buying signals for more trade opportunities.
        """
        # Improved dip detection (more sensitive)
        sharp_pct_drop = df["pct_change"] < -self.dip_drop_pct
        sharp_atr_drop = df["pct_change"] < -(self.dip_drop_atr_mult * df["atr"] / df["close"].shift(1))
        
        # Consecutive down days for stronger signals
        two_down_days = (df["pct_change"] < 0) & (df["pct_change"].shift(1) < 0)
        
        # Check for consolidation after drops
        range_N = df["high"].rolling(self.dip_consol_window).max() - df["low"].rolling(self.dip_consol_window).min()
        consol = range_N < (self.dip_consol_atr_mult * df["atr"])
        
        # RSI oversold condition for better timing
        rsi_oversold = df["rsi"] < 30
        
        # Dip buy signals with different strengths
        df["dip_buy_strong"] = (sharp_pct_drop.shift(1) | sharp_atr_drop.shift(1)) & consol & rsi_oversold
        df["dip_buy_normal"] = (sharp_pct_drop.shift(1) | sharp_atr_drop.shift(1)) & consol
        df["dip_buy_weak"] = two_down_days & (df["close"] < df["ema20"]) & (df["rsi"] < 40)
        
        # Combined dip buying signal
        df["dip_buy"] = df["dip_buy_strong"] | df["dip_buy_normal"]
        
        # Add dip buy signal for ranging markets (more conservative)
        df["range_dip_buy"] = (df["market_regime"] == "ranging") & (df["close"] < df["bb_lower"]) & (df["rsi"] < 30)
        
        return df
    
    def add_hft_components(self, df):
        """
        Add HFT-inspired components for quicker entries and exits.
        """
        # Optimal value band for fair price estimation
        df["value_middle"] = df["keltner_middle"]
        
        # Optimal entry/exit bands (volatility-adjusted)
        entry_band_width = df["atr"] * 0.5
        df["optimal_buy_zone"] = df["value_middle"] - entry_band_width
        df["optimal_sell_zone"] = df["value_middle"] + entry_band_width
        
        # Short-term mean reversion signals
        df["mean_reversion_buy"] = (
            (df["close"] < df["bb_lower"]).fillna(False) & 
            (df["rsi_fast"] < 20).fillna(False) &
            (df["pct_change_3"] < -1.5 * df["atr_pct"]).fillna(False)
        )
        
        df["mean_reversion_sell"] = (
            (df["close"] > df["bb_upper"]).fillna(False) & 
            (df["rsi_fast"] > 80).fillna(False) &
            (df["pct_change_3"] > 1.5 * df["atr_pct"]).fillna(False)
        )
        
        # Momentum burst signals (very short-term momentum)
        df["momentum_burst_up"] = (df["close"] > df["open"]) & (df["close"] > df["high"].shift(1)) & (df["volume"] > df["volume"].shift(1) * 1.2)
        df["momentum_burst_down"] = (df["close"] < df["open"]) & (df["close"] < df["low"].shift(1)) & (df["volume"] > df["volume"].shift(1) * 1.2)
        
        # Ultra-fast trend change detection
        df["ultra_trend_change_up"] = (df["ema3"].shift(1) < df["ema8"].shift(1)) & (df["ema3"] > df["ema8"])
        df["ultra_trend_change_down"] = (df["ema3"].shift(1) > df["ema8"].shift(1)) & (df["ema3"] < df["ema8"])
        
        # Calculate profit targets for faster exits - initialize with 0 instead of NaN
        df["profit_target_long"] = 0.0
        df["profit_target_short"] = 0.0
        df["partial_profit_long"] = 0.0
        df["partial_profit_short"] = 0.0
        
        # Time since last trade placeholder
        df["bars_since_last_trade"] = 0
        
        return df

    def generate_signals(self, df, equity=10000.0):
        """
        Generate optimized trading signals with advanced risk management.
        """
        df["trade_type"] = TradeType.HOLD.value
        df["Position"] = 0
        df["entry_price"] = 0.0  # Initialize with 0 instead of NaN
        df["stoploss_price"] = 0.0  # Initialize with 0 instead of NaN
        df["position_size"] = 0.0
        df["bars_since_last_trade"] = 0  # Initialize with 0 instead of NaN
        df["trailing_stop"] = 0.0  # Initialize with 0 instead of NaN
        df["time_in_trade"] = 0
        df["partial_profit_taken"] = False
        df["trade_risk_pct"] = 0.0
        
        # Equity curve simulation - fix by ensuring equity is a value, not a reference to column
        initial_equity = float(equity)  # Make a copy of the input equity value
        df["equity"] = initial_equity
        df["drawdown_pct"] = 0.0
        
        # Initialize max equity tracking
        self.max_reached_equity = initial_equity
        
        # Track time since last trade
        last_trade_idx = -999
        
        for i in range(len(df)):
            df.loc[i, "bars_since_last_trade"] = i - last_trade_idx
            
            # Simulate equity curve and update drawdown metrics
            if i > 0:
                prev_equity = float(df.loc[i-1, "equity"])  # Get as float to avoid reference issues
                prev_position = df.loc[i-1, "Position"]
                
                if prev_position != 0:
                    # Calculate P&L for the position
                    entry_price = df.loc[i-1, "entry_price"]
                    position_size = df.loc[i-1, "position_size"]
                    
                    # Add NaN handling
                    if pd.isna(entry_price) or pd.isna(position_size):
                        entry_price = 0.0 if pd.isna(entry_price) else entry_price
                        position_size = 0.0 if pd.isna(position_size) else position_size
                    
                    if prev_position == 1:  # Long position
                        pnl = (df.loc[i, "close"] - df.loc[i-1, "close"]) * position_size
                    else:  # Short position
                        pnl = (df.loc[i-1, "close"] - df.loc[i, "close"]) * position_size
                    
                    # Update equity - ensure we're using numeric values
                    df.loc[i, "equity"] = prev_equity + float(0.0 if pd.isna(pnl) else pnl)
                else:
                    df.loc[i, "equity"] = prev_equity
                
                # Update max reached equity
                self.max_reached_equity = max(self.max_reached_equity, float(df.loc[i, "equity"]))
                
                # Calculate drawdown
                if self.max_reached_equity > 0:  # Prevent division by zero
                    df.loc[i, "drawdown_pct"] = (self.max_reached_equity - df.loc[i, "equity"]) / self.max_reached_equity * 100
                else:
                    df.loc[i, "drawdown_pct"] = 0.0
                
                self.current_drawdown = df.loc[i, "drawdown_pct"]
                
                # Check if in recovery mode
                self.in_recovery_mode = self.current_drawdown > self.recovery_mode_drawdown_pct
                
                # Update time in trade for active positions
                if prev_position != 0:
                    df.loc[i, "time_in_trade"] = df.loc[i-1, "time_in_trade"] + 1
                    
                    # Update trailing stop for active positions with NaN handling
                    if prev_position == 1:  # Long position
                        new_trail = df.loc[i, "close"] - (df.loc[i, "atr"] * self.trailing_stop_atr_multiple)
                        prev_trailing_stop = df.loc[i-1, "trailing_stop"]
                        prev_trailing_stop = 0.0 if pd.isna(prev_trailing_stop) else prev_trailing_stop
                        
                        if prev_trailing_stop == 0.0 or new_trail > prev_trailing_stop:
                            df.loc[i, "trailing_stop"] = new_trail
                        else:
                            df.loc[i, "trailing_stop"] = prev_trailing_stop
                    else:  # Short position
                        new_trail = df.loc[i, "close"] + (df.loc[i, "atr"] * self.trailing_stop_atr_multiple)
                        prev_trailing_stop = df.loc[i-1, "trailing_stop"]
                        prev_trailing_stop = 0.0 if pd.isna(prev_trailing_stop) else prev_trailing_stop
                        
                        if prev_trailing_stop == 0.0 or new_trail < prev_trailing_stop:
                            df.loc[i, "trailing_stop"] = new_trail
                        else:
                            df.loc[i, "trailing_stop"] = prev_trailing_stop
                    
                    # Check for partial profit triggers with robust NaN handling
                    partial_profit_long = df.loc[i-1, "partial_profit_long"]
                    partial_profit_short = df.loc[i-1, "partial_profit_short"]
                    
                    # Convert NaN to 0 to prevent NaN comparison issues
                    partial_profit_long = 0.0 if pd.isna(partial_profit_long) else partial_profit_long
                    partial_profit_short = 0.0 if pd.isna(partial_profit_short) else partial_profit_short
                    
                    if prev_position == 1 and not df.loc[i-1, "partial_profit_taken"] and \
                       partial_profit_long > 0 and df.loc[i, "close"] >= partial_profit_long:
                        # Take partial profits on longs
                        df.loc[i, "partial_profit_taken"] = True
                        df.loc[i, "position_size"] = df.loc[i-1, "position_size"] * 0.5
                    
                    elif prev_position == -1 and not df.loc[i-1, "partial_profit_taken"] and \
                         partial_profit_short > 0 and df.loc[i, "close"] <= partial_profit_short:
                        # Take partial profits on shorts
                        df.loc[i, "partial_profit_taken"] = True
                        df.loc[i, "position_size"] = df.loc[i-1, "position_size"] * 0.5
                    else:
                        df.loc[i, "partial_profit_taken"] = df.loc[i-1, "partial_profit_taken"]
                        df.loc[i, "position_size"] = df.loc[i-1, "position_size"]
                else:
                    df.loc[i, "partial_profit_taken"] = False
                    
            # Forward fill entry price and stop loss
            if i > 0 and df.loc[i-1, "Position"] != 0:
                df.loc[i, "entry_price"] = df.loc[i-1, "entry_price"]
                df.loc[i, "stoploss_price"] = df.loc[i-1, "stoploss_price"]
                
                # Ensure no NaN values propagate
                if pd.isna(df.loc[i, "entry_price"]): df.loc[i, "entry_price"] = 0.0
                if pd.isna(df.loc[i, "stoploss_price"]): df.loc[i, "stoploss_price"] = 0.0
        
        # Apply risk-adjusted max allowed drawdown
        risk_factor = 1.0
        if self.in_recovery_mode:
            risk_factor = self.recovery_risk_factor
        
        # === ENTRY CONDITIONS WITH INDEX ALIGNMENT FIX ===
        
        # Dynamic position sizing based on volatility and drawdown
        vol_adjusted_risk = self.max_risk_per_trade_pct * df["vol_pos_multiplier"].fillna(1.0) * risk_factor
        
        # Create bullish entry conditions with explicit index alignment
        # First create all individual conditions
        strong_bull_trend = ((df["market_regime"] == "strong_bull") & df["trend_aligned_bull"])
        weak_bull_confirm = ((df["market_regime"] == "weak_bull") & df["fast_aligned_bull"] & (df["rsi"] < 60))
        dip_buy_bull = ((df["market_regime"] == "strong_bull") & df["dip_buy"])
        breakout_up = df["high_prob_breakout_up"]
        mean_rev_bull = (df["market_regime"] == "strong_bull") & df["mean_reversion_buy"]
        range_dip = ((df["market_regime"] == "ranging") & df["range_dip_buy"] & (df["ema8"] > df["ema20"]))
        
        # Combine entry conditions with OR
        bull_conditions = (strong_bull_trend | weak_bull_confirm | dip_buy_bull | 
                          breakout_up | mean_rev_bull | range_dip)
        
        # Apply common filters
        no_position = (df["Position"].shift().fillna(0) == 0)
        min_bars_passed = (df["bars_since_last_trade"] >= self.min_trade_interval)
        recovery_filter = (~self.in_recovery_mode | 
                          (self.in_recovery_mode & df["market_regime"].isin(["strong_bull", "strong_bear"])))
        
        # Combine all conditions with AND
        bullish_entry = bull_conditions & no_position & min_bars_passed & recovery_filter
        
        # Same approach for bearish conditions
        strong_bear_trend = ((df["market_regime"] == "strong_bear") & df["trend_aligned_bear"])
        weak_bear_confirm = ((df["market_regime"] == "weak_bear") & df["fast_aligned_bear"] & (df["rsi"] > 40))
        breakout_down = df["high_prob_breakout_down"]
        mean_rev_bear = (df["market_regime"] == "strong_bear") & df["mean_reversion_sell"]
        
        # Combine bearish conditions
        bear_conditions = (strong_bear_trend | weak_bear_confirm | breakout_down | mean_rev_bear)
        bearish_entry = bear_conditions & no_position & min_bars_passed & recovery_filter
        
        # === APPLY ENTRY SIGNALS WITH INDEX ALIGNMENT FIX ===
        
        # Process bullish entries while ensuring index alignment
        if bullish_entry.any():
            # Get the indices where bullish_entry is True
            bull_indices = df.index[bullish_entry]
            
            # Apply position and entry price updates
            df.loc[bull_indices, "Position"] = 1
            df.loc[bull_indices, "entry_price"] = df.loc[bull_indices, "close"]
            
            # Set tiered stops based on market regime - with explicit indexing
            strong_bull_indices = bull_indices[df.loc[bull_indices, "market_regime"].isin(["strong_bull"])]
            if len(strong_bull_indices) > 0:
                df.loc[strong_bull_indices, "stoploss_price"] = (
                    df.loc[strong_bull_indices, "entry_price"] - 
                    df.loc[strong_bull_indices, "atr"] * self.atr_multiplier
                )
            
            other_bull_indices = bull_indices[~df.loc[bull_indices, "market_regime"].isin(["strong_bull"])]
            if len(other_bull_indices) > 0:
                df.loc[other_bull_indices, "stoploss_price"] = (
                    df.loc[other_bull_indices, "entry_price"] - 
                    df.loc[other_bull_indices, "atr"] * (self.atr_multiplier * 0.8)
                )
            
            # Calculate risk per share and position sizes
            df.loc[bull_indices, "trade_risk_pct"] = vol_adjusted_risk.loc[bull_indices]
            risk_per_share = (df.loc[bull_indices, "entry_price"] - df.loc[bull_indices, "stoploss_price"]).abs().fillna(0)
            
            # Calculate position sizes
            for idx in bull_indices:
                df.loc[idx, "position_size"] = self.calculate_position_size(
                    df.loc[idx, "equity"],
                    df.loc[idx, "entry_price"],
                    df.loc[idx, "stoploss_price"],
                    df.loc[idx, "trade_risk_pct"]
                )
            
            # Calculate profit targets
            valid_risk_indices = bull_indices[risk_per_share[bull_indices] > 0]
            if len(valid_risk_indices) > 0:
                for idx in valid_risk_indices:
                    # Use np.abs() instead of .abs() method for NumPy float values
                    risk = np.abs(df.loc[idx, "entry_price"] - df.loc[idx, "stoploss_price"])
                    df.loc[idx, "profit_target_long"] = df.loc[idx, "entry_price"] + (risk * self.profit_target_r_multiple)
                    df.loc[idx, "partial_profit_long"] = df.loc[idx, "entry_price"] + (risk * self.partial_exit_r_multiple)
            
            # Set trailing stops
            df.loc[bull_indices, "trailing_stop"] = df.loc[bull_indices, "stoploss_price"]
            
            # Update last trade index
            for idx in bull_indices:
                last_trade_idx = idx
                indices_to_update = df.index[df.index > idx]
                if len(indices_to_update) > 0:
                    df.loc[indices_to_update, "bars_since_last_trade"] = indices_to_update - idx
        
        # Similarly process bearish entries
        if bearish_entry.any():
            # Get the indices where bearish_entry is True
            bear_indices = df.index[bearish_entry]
            
            # Apply position and entry price updates
            df.loc[bear_indices, "Position"] = -1
            df.loc[bear_indices, "entry_price"] = df.loc[bear_indices, "close"]
            
            # Set tiered stops based on market regime - with explicit indexing
            strong_bear_indices = bear_indices[df.loc[bear_indices, "market_regime"].isin(["strong_bear"])]
            if len(strong_bear_indices) > 0:
                df.loc[strong_bear_indices, "stoploss_price"] = (
                    df.loc[strong_bear_indices, "entry_price"] + 
                    df.loc[strong_bear_indices, "atr"] * self.atr_multiplier
                )
            
            other_bear_indices = bear_indices[~df.loc[bear_indices, "market_regime"].isin(["strong_bear"])]
            if len(other_bear_indices) > 0:
                df.loc[other_bear_indices, "stoploss_price"] = (
                    df.loc[other_bear_indices, "entry_price"] + 
                    df.loc[other_bear_indices, "atr"] * (self.atr_multiplier * 0.8)
                )
            
            # Calculate risk per share and position sizes
            df.loc[bear_indices, "trade_risk_pct"] = vol_adjusted_risk.loc[bear_indices]
            risk_per_share = (df.loc[bear_indices, "entry_price"] - df.loc[bear_indices, "stoploss_price"]).abs().fillna(0)
            
            # Calculate position sizes
            for idx in bear_indices:
                df.loc[idx, "position_size"] = self.calculate_position_size(
                    df.loc[idx, "equity"],
                    df.loc[idx, "entry_price"],
                    df.loc[idx, "stoploss_price"],
                    df.loc[idx, "trade_risk_pct"]
                )
            
            # Calculate profit targets
            valid_risk_indices = bear_indices[risk_per_share[bear_indices] > 0]
            if len(valid_risk_indices) > 0:
                for idx in valid_risk_indices:
                    # Use np.abs() instead of .abs() method for NumPy float values
                    risk = np.abs(df.loc[idx, "entry_price"] - df.loc[idx, "stoploss_price"])
                    df.loc[idx, "profit_target_short"] = df.loc[idx, "entry_price"] - (risk * self.profit_target_r_multiple)
                    df.loc[idx, "partial_profit_short"] = df.loc[idx, "entry_price"] - (risk * self.partial_exit_r_multiple)
            
            # Set trailing stops
            df.loc[bear_indices, "trailing_stop"] = df.loc[bear_indices, "stoploss_price"]
            
            # Update last trade index
            for idx in bear_indices:
                last_trade_idx = idx
                indices_to_update = df.index[df.index > idx]
                if len(indices_to_update) > 0:
                    df.loc[indices_to_update, "bars_since_last_trade"] = indices_to_update - idx
        
        # Handle exits using same index-based approach
        exit_long = (
            (df["Position"].shift().fillna(0) == 1) & 
            (
                # Major trend change
                (df["ema8"] < df["ema20"]).fillna(False) |
                # Trend regime shift
                (~df["market_regime"].isin(["strong_bull", "weak_bull"])) |
                # Extreme RSI
                (df["rsi"] > self.rsi_overbought).fillna(False) |
                # Ultra fast trend change
                df["ultra_trend_change_down"].fillna(False) |
                # Take full profit at target - ensure no NaN issues
                ((df["close"] >= df["profit_target_long"]) & (df["profit_target_long"] > 0)).fillna(False) |
                # Time-based stop
                (df["time_in_trade"] > self.time_stop_bars) |
                # Trailing stop hit - ensure no NaN issues
                ((df["low"] < df["trailing_stop"]) & (df["trailing_stop"] > 0)).fillna(False)
            )
        )
        
        exit_short = (
            (df["Position"].shift().fillna(0) == -1) & 
            (
                # Major trend change
                (df["ema8"] > df["ema20"]).fillna(False) |
                # Trend regime shift
                (~df["market_regime"].isin(["strong_bear", "weak_bear"])) |
                # Extreme RSI
                (df["rsi"] < self.rsi_oversold).fillna(False) |
                # Ultra fast trend change
                df["ultra_trend_change_up"].fillna(False) |
                # Take full profit at target - ensure no NaN issues
                ((df["close"] <= df["profit_target_short"]) & (df["profit_target_short"] > 0)).fillna(False) |
                # Time-based stop
                (df["time_in_trade"] > self.time_stop_bars) |
                # Trailing stop hit - ensure no NaN issues
                ((df["high"] > df["trailing_stop"]) & (df["trailing_stop"] > 0)).fillna(False)
            )
        )
        
        long_stoploss = (
            (df["Position"].shift().fillna(0) == 1) & 
            ((df["low"] < df["stoploss_price"].shift()) & (df["stoploss_price"].shift() > 0)).fillna(False)
        )
        
        short_stoploss = (
            (df["Position"].shift().fillna(0) == -1) & 
            ((df["high"] > df["stoploss_price"].shift()) & (df["stoploss_price"].shift() > 0)).fillna(False)
        )
        
        exit_long_condition = exit_long | long_stoploss
        exit_short_condition = exit_short | short_stoploss
        
        exit_condition = exit_long_condition | exit_short_condition
        if not exit_condition.empty and exit_condition.any():
            df.loc[exit_condition, "Position"] = 0
            df.loc[exit_condition, "entry_price"] = 0.0  # Use 0 instead of NaN
            df.loc[exit_condition, "stoploss_price"] = 0.0  # Use 0 instead of NaN
            df.loc[exit_condition, "profit_target_long"] = 0.0  # Reset profit targets
            df.loc[exit_condition, "profit_target_short"] = 0.0
            df.loc[exit_condition, "partial_profit_long"] = 0.0
            df.loc[exit_condition, "partial_profit_short"] = 0.0
            df.loc[exit_condition, "position_size"] = 0.0
            df.loc[exit_condition, "time_in_trade"] = 0
            df.loc[exit_condition, "trailing_stop"] = 0.0  # Use 0 instead of NaN
            df.loc[exit_condition, "partial_profit_taken"] = False
            
            # Update last trade index
            for idx in exit_condition[exit_condition].index:
                last_trade_idx = idx
                indices_to_update = df.index[df.index > idx]
                if len(indices_to_update) > 0:
                    df.loc[indices_to_update, "bars_since_last_trade"] = indices_to_update - idx
        
        # Forward-fill necessary position data (ensure partial profits are applied)
        df["Position"] = df["Position"].ffill().fillna(0)
        
        # Ensure no NaN values in any critical columns
        numeric_columns = [
            "entry_price", "stoploss_price", "position_size", "trailing_stop",
            "profit_target_long", "profit_target_short", 
            "partial_profit_long", "partial_profit_short"
        ]
        
        for col in numeric_columns:
            df[col] = df[col].fillna(0)
        
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

    def calculate_position_size(self, equity, entry_price, stoploss_price, risk_pct=None):
        """
        Calculate position size based on risk percentage with volatility adjustment.
        """
        # Default to standard risk if not specified
        if risk_pct is None:
            risk_pct = self.risk_pct
            
        # Handle the case where inputs are pandas Series
        if isinstance(entry_price, pd.Series) and isinstance(stoploss_price, pd.Series):
            # Fill NaN values with zeros to prevent calculations with NaN
            entry_price = entry_price.fillna(0)
            stoploss_price = stoploss_price.fillna(0)
            
            # If equity is a Series, ensure it's float to avoid reference issues
            if isinstance(equity, pd.Series):
                equity = equity.astype(float).fillna(0)
            else:
                # Otherwise create a Series with proper values
                equity = pd.Series(float(equity), index=entry_price.index)
                
            # Calculate risk amount for each trade
            risk_amount = equity * (risk_pct / 100)
            risk_per_unit = (entry_price - stoploss_price).abs()
            
            # Calculate position size only where risk_per_unit is valid
            position_size = pd.Series(0.0, index=risk_per_unit.index)
            valid_mask = (risk_per_unit > 0)
            position_size.loc[valid_mask] = risk_amount.loc[valid_mask] / risk_per_unit.loc[valid_mask]
            
            return position_size.round(2)
        else:
            # Handle scalar NaN values
            if pd.isna(entry_price) or pd.isna(stoploss_price):
                return 0
                
            # Scalar calculation
            risk_amount = float(equity) * (risk_pct / 100)
            risk_per_unit = abs(entry_price - stoploss_price)
            
            # Avoid division by zero or NaN
            if risk_per_unit <= 0:
                return 0
                
            position_size = risk_amount / risk_per_unit
            return np.round(position_size, 2)
