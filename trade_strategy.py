import pandas as pd
import numpy as np
from cooked import Strategy

class TradingStrategy(Strategy):
    def __init__(self, fast_window=20, slow_window=50, risk_pct=2):
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.risk_pct = risk_pct
        
    def generate_signals(self, data):
        """Generate trading signals based on moving average crossover."""
        df = data.copy()
        
        # Calculate moving averages
        df['MA_fast'] = df['close'].rolling(window=self.fast_window).mean()
        df['MA_slow'] = df['close'].rolling(window=self.slow_window).mean()
        
        # Initialize signals
        df['signal'] = 0
        
        # Generate buy signal when fast MA crosses above slow MA
        df.loc[df['MA_fast'] > df['MA_slow'], 'signal'] = 1
        
        # Generate sell signal when fast MA crosses below slow MA
        df.loc[df['MA_fast'] < df['MA_slow'], 'signal'] = -1
        
        return df
    
    def calculate_stop_loss(self, entry_price, direction='long'):
        """Calculate stop loss level for a trade."""
        stop_loss_pct = 0.02  # 2% stop loss
        
        if direction == 'long':
            return entry_price * (1 - stop_loss_pct)
        else:  # short
            return entry_price * (1 + stop_loss_pct)
