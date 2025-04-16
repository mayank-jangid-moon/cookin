import pandas as pd
import numpy as np
import talib as ta
from enum import Enum
import math


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
    Ichimoku Cloud based trading strategy with multiple indicator confirmation.
    
    This strategy generates trading signals by combining:
    1. Ichimoku Cloud indicators
    2. MACD for trend confirmation
    3. RSI for overbought/oversold conditions
    4. Bollinger Bands for volatility
    5. OBV for volume confirmation
    6. ATR for volatility and risk management
    
    The strategy implements regime switching based on market volatility
    and uses comprehensive risk management techniques.
    """
    
    def __init__(self, 
                 take_profit=10, 
                 take_profit2=10,
                 stop_loss=100,
                 stop_loss2=100,
                 atr_multiplier=1.0,
                 atr_multiplier2=10.0,
                 max_drawdown_limit=15,
                 max_drawdown_limit2=100,
                 change_limit=7,
                 daily_change_limit=7,
                 low_high_change_limit=100,
                 rsi_period=14):
        """
        Initialize the strategy with customizable parameters matching the notebook.
        
        Parameters:
            take_profit (float): Percentage for trailing take profit (default: 10%)
            take_profit2 (float): Percentage for normal take profit (default: 10%)
            stop_loss (float): Percentage for trailing stop loss (default: 100% - not used)
            stop_loss2 (float): Percentage for normal stop loss (default: 100% - not used)
            atr_multiplier (float): Multiplier for ATR stop loss (default: 1.0)
            atr_multiplier2 (float): Multiplier for ATR take profit (default: 10.0 - not used)
            max_drawdown_limit (float): Maximum drawdown limit to exit a trade (default: 15%)
            max_drawdown_limit2 (float): Max drawdown limit2 (default: 100% - not used)
            change_limit (float): Percentage limit for intraday price change (default: 7%)
            daily_change_limit (float): Percentage limit for daily price change (default: 7%)
            low_high_change_limit (float): Percentage limit for low-high change (default: 100% - not used)
            rsi_period (int): Period for RSI calculation (default: 14)
        """
        # Strategy parameters
        self.take_profit = take_profit  # Percentage for trailing take profit
        self.take_profit2 = take_profit2  # Percentage for normal take profit
        self.stop_loss = stop_loss  # Percentage for trailing stop loss (not used in original)
        self.stop_loss2 = stop_loss2  # Percentage for normal stop loss (not used in original)
        self.atr_multiplier = atr_multiplier  # Multiplier for ATR stop loss
        self.atr_multiplier2 = atr_multiplier2  # Multiplier for ATR take profit (not used in original)
        self.max_drawdown_limit = max_drawdown_limit  # Maximum drawdown limit to exit a trade
        self.max_drawdown_limit2 = max_drawdown_limit2  # Max drawdown limit2 (not used in original)
        self.change_limit = change_limit  # Percentage limit for intraday price change
        self.daily_change_limit = daily_change_limit  # Percentage limit for daily price change
        self.low_high_change_limit = low_high_change_limit  # Percentage limit for low-high change (not used in original)
        self.rsi_period = rsi_period  # Period for RSI calculation
        
    def run(self, df: pd.DataFrame, equity: float = 1000000.0) -> pd.DataFrame:
        """
        Execute the strategy on the provided price data.
        
        Parameters:
            df (pd.DataFrame): Price data containing 'open', 'high', 'low', 'close', 'volume' columns
            equity (float): Starting account equity for position sizing calculations (default: $1,000,000)
            
        Returns:
            pd.DataFrame: The input dataframe augmented with strategy indicators and signals
        """
        # Rename columns to match expected format
        data = df.copy()
        
        # Ensure we have the correct column names for both lowercase and uppercase
        for src, dst in [('open', 'Open'), ('high', 'High'), ('low', 'Low'), ('close', 'Close')]:
            if src in data.columns and dst not in data.columns:
                data[dst] = data[src]
            elif dst in data.columns and src not in data.columns:
                data[src] = data[dst]
                
        # Calculate indicators using the exact same functions as the notebook
        data = self.heikin_ashi_candlesticks(data)
        data = self.calculate_rsi(data)
        data = self.calculate_macd(data)
        data = self.calculate_ichimoku_cloud(data)
        data = self.calculate_bollinger_bands(data)
        data = self.calculate_obv(data)
        data = self.calculate_obv2(data)
        data = self.generate_atr_signals(data)
        data = self.volume_indicator(data)
        data = self.ichimoku_cloud(data)
        
        # Generate signals exactly like the notebook
        data = self.generate_signals(data)
        
        # Initialize columns for position tracking
        data["trade_type"] = TradeType.HOLD.value
        data["position_size"] = 0.0
        
        # Create portfolio tracking dataframe
        portfolio_df = pd.DataFrame()
        portfolio_df['No_of_Stocks'] = 0
        portfolio_df['Portfolio_Value'] = 0
        portfolio_df['Profit_From_Initial_Capital'] = 0
        portfolio_df['Current_Position'] = 0
        
        # Apply risk management and generate final signals
        data = self.calculate_with_risk_management(data, portfolio_df, equity)
        
        return data

    def heikin_ashi_candlesticks(self, data):
        """Calculate Heikin Ashi candles to de-noise data exactly as in notebook"""
        # Make a copy to avoid warnings
        df = data.copy()
        
        # Calculate HA_CLOSE
        df["HA_CLOSE"] = (df["Low"] + df["High"] + df["Close"] + df["Open"]) / 4
        
        # Pre-allocate HA_OPEN
        df["HA_OPEN"] = df["Open"].copy()
        
        # Calculate HA_OPEN
        for i in range(len(df)):
            if i == 0:
                # For the first row
                df.loc[i, "HA_OPEN"] = (df.loc[i, "Open"] + df.loc[i, "Close"]) / 2
            else:
                # For subsequent rows
                df.loc[i, "HA_OPEN"] = (df.loc[i-1, "HA_OPEN"] + df.loc[i-1, "HA_CLOSE"]) / 2
        
        # Calculate HA_HIGH and HA_LOW
        df["HA_HIGH"] = df[["HA_OPEN", "HA_CLOSE", "High"]].max(axis=1)
        df["HA_LOW"] = df[["HA_OPEN", "HA_CLOSE", "Low"]].min(axis=1)
        df['close_denoised'] = df['Close']  # Use close as denoised in notebook
        
        return df

    def calculate_ichimoku_cloud(self, data):
        """Calculate Ichimoku Cloud indicators with smaller rolling window"""
        data['Tenkan-sen'] = (data['High'].rolling(window=9).max() + data['Low'].rolling(window=9).min()) / 2
        data['Kijun-sen'] = (data['High'].rolling(window=12).max() + data['Low'].rolling(window=12).min()) / 2
        data['Senkou Span A'] = ((data['Tenkan-sen'] + data['Kijun-sen']) / 2).shift(12)
        data['Senkou Span B'] = ((data['High'].rolling(window=26).max() + data['Low'].rolling(window=26).min()) / 2).shift(12)
        return data

    def ichimoku_cloud(self, data):
        """Calculate Ichimoku Cloud indicators with longer rolling window - match notebook exactly"""
        high_9 = data['High'].rolling(window=9).max()
        low_9 = data['Low'].rolling(window=9).min()
        data['CL'] = (high_9 + low_9) / 2
        
        high_30 = data['High'].rolling(window=30).max()
        low_30 = data['Low'].rolling(window=30).min()
        data['BL'] = (high_30 + low_30) / 2
        
        data['SA'] = ((data['CL'] + data['BL']) / 2).shift(26)
        
        high_58 = data['High'].rolling(window=58).max()
        low_58 = data['Low'].rolling(window=58).min()
        data['SB'] = ((high_58 + low_58) / 2).shift(26)
        
        return data

    def volume_indicator(self, data):
        """Calculate volume indicator to identify abstract change in volume - match notebook exactly"""
        data['pc'] = ((data['Close'].diff())/data['Close'].shift(1))*100
        data['volume_change'] = abs(data['volume'].diff())
        data['vc_expo_fast'] = data['volume_change'].ewm(span=5, adjust=False).mean()
        data['fast_c'] = abs(data['vc_expo_fast'].diff())
        data['vc_expo_slow'] = data['volume_change'].ewm(span=14, adjust=False).mean()
        data['slow_c'] = abs(data['vc_expo_slow'].diff())
        data['ratio'] = data['vc_expo_fast']/data['vc_expo_slow']
        data['diff'] = abs(data['ratio'].diff())
        return data

    def calculate_macd(self, data):
        """Calculate MACD indicator - match notebook exactly"""
        data['EMA26'] = data['close_denoised'].ewm(span=26, adjust=False).mean()
        data['EMA12'] = data['close_denoised'].ewm(span=12, adjust=False).mean()
        data['MACD Line'] = data['EMA12'] - data['EMA26']
        data['Signal Line'] = data['MACD Line'].ewm(span=8, adjust=False).mean()
        return data

    def calculate_rsi(self, data, window=14):
        """Calculate RSI indicator - match notebook exactly"""
        delta = data['close_denoised'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(span=window, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(span=window, adjust=False).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        data['RSI'] = rsi
        return data

    def calculate_bollinger_bands(self, data, window=20, num_std=1.8):
        """Calculate Bollinger Bands indicator - match notebook exactly"""
        data['Middle'] = data['close_denoised'].rolling(window=window).mean()
        data['Upper'] = data['Middle'] + num_std * data['close_denoised'].rolling(window=window).std()
        data['Lower'] = data['Middle'] - num_std * data['close_denoised'].rolling(window=window).std()
        return data

    def calculate_obv(self, data):
        """Calculate On Balance Volume indicator - match notebook exactly"""
        data['price_change'] = data['close_denoised'].diff()
        data['direction'] = np.where(data['price_change'] > 0, 1, -1)
        data['volume_direction'] = data['direction'] * data['volume']
        data['OBV'] = data['volume_direction'].cumsum()
        data.drop(['price_change', 'direction', 'volume_direction'], axis=1, inplace=True)
        return data

    def calculate_obv2(self, data):
        """Calculate OBV with exponential average - match notebook exactly"""
        obv = [0]
        for i in range(1, len(data)):
            if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
                obv.append(obv[-1] + data['volume'].iloc[i])
            elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
                obv.append(obv[-1] - data['volume'].iloc[i])
            else:
                obv.append(obv[-1])
        
        data['obv'] = obv
        data['e_avg'] = data['obv'].ewm(span=12, adjust=True).mean()
        period = 12
        data['slope'] = (data['e_avg'] - data['e_avg'].shift(period)) / period
        return data

    def generate_atr_signals(self, data):
        """Calculate Average True Range indicator - match notebook exactly"""
        atr_window = 14
        data['ATR'] = data['High'].rolling(window=atr_window).apply(lambda x: max(x) - min(x), raw=True)
        
        multiplier = 2.0
        data['ATR_Stop_Loss'] = data['Close'] - multiplier * data['ATR']
        data['ATR_Take_Profit'] = data['Close'] + multiplier * data['ATR']
        return data

    def generate_signals(self, data):
        """Generate trading signals based on the indicator calculations - exact notebook logic"""
        data['signals'] = 0
        
        # Calculate historical values for reference
        data['Historical_OBV'] = data['OBV'].rolling(window=55).mean()
        data['MACD'] = data['MACD Line'] - data['Signal Line']
        data['Historical_ATR'] = data['ATR'].rolling(window=55).mean()
        data['Historical_Volume'] = data['volume'].rolling(window=26).mean()
        
        for i in range(len(data)):
            # Long entry conditions
            if (data['close_denoised'].iloc[i] >= data['Upper'].iloc[i] and
                data['ATR'].iloc[i] >= data['Historical_ATR'].iloc[i] or
                data['RSI'].iloc[i] >= 80 and
                data['MACD Line'].iloc[i] >= data['Signal Line'].iloc[i] and
                data['volume'].iloc[i] >= data['Historical_Volume'].iloc[i]):
                data.loc[i, 'signals'] = 1

            elif (data['close_denoised'].iloc[i] >= data['Senkou Span A'].iloc[i] and
                  data['close_denoised'].iloc[i] >= data['Senkou Span B'].iloc[i] and
                  data['OBV'].iloc[i] >= data['Historical_OBV'].iloc[i] and
                  data['MACD'].iloc[i] >= 0 and
                  data['volume'].iloc[i] >= data['Historical_Volume'].iloc[i]):
                data.loc[i, 'signals'] = 1

            elif ((data['diff'].iloc[i] >= 0.2 and data['obv'].iloc[i] > data['e_avg'].iloc[i]) or 
                  (data['diff'].iloc[i] >= 0.2 and 
                   (data['Close'].iloc[i] > data['SA'].iloc[i]) and
                   data['Close'].iloc[i] > data['SB'].iloc[i]) and 
                   (data['CL'].iloc[i] > data['BL'].iloc[i])):
                data.loc[i, 'signals'] = 1

            # Short entry conditions
            elif (data['close_denoised'].iloc[i] <= data['Lower'].iloc[i] and
                  data['ATR'].iloc[i] >= data['Historical_ATR'].iloc[i] or
                  data['RSI'].iloc[i] >= 80 and
                  data['MACD Line'].iloc[i] <= data['Signal Line'].iloc[i] and
                  data['volume'].iloc[i] >= data['Historical_Volume'].iloc[i]):
                data.loc[i, 'signals'] = -1

            elif (data['close_denoised'].iloc[i] <= data['Senkou Span A'].iloc[i] and
                  data['close_denoised'].iloc[i] <= data['Senkou Span B'].iloc[i] and
                  data['OBV'].iloc[i] <= data['Historical_OBV'].iloc[i] and
                  data['MACD'].iloc[i] <= 0 and
                  data['volume'].iloc[i] >= data['Historical_Volume'].iloc[i]):
                data.loc[i, 'signals'] = -1

            elif ((data['diff'].iloc[i] >= 0.2 and data['obv'].iloc[i] < data['e_avg'].iloc[i]) or 
                  (data['diff'].iloc[i] >= 0.2 and 
                   (data['Close'].iloc[i] < data['SA'].iloc[i]) and
                   data['Close'].iloc[i] < data['SB'].iloc[i]) and 
                   (data['CL'].iloc[i] < data['BL'].iloc[i])):
                data.loc[i, 'signals'] = -1
        
        return data
    
    def calculate_with_risk_management(self, data, dd1, initial_balance=1000000):
        """
        Implementation of the risk management and final position processing
        This closely follows the 'calculate' function in the notebook
        """
        # Initialize position and trade tracking
        data['Position'] = 0
        data['entry_price'] = 0.0
        data['stoploss_price'] = 0.0
        data['trailing_sl'] = 0.0
        data['trailing_tp'] = 0.0
        
        # Track positions
        current_position = 0
        entry_index = 0
        
        # For risk management calculation
        max_drawdown = []
        balance = initial_balance
        capital = initial_balance
        dd2 = []
        
        # For portfolio value calculation
        num_stocks = []
        remains = []
        
        # For risk management
        tsll = []
        ttpl = []
        drawdown2 = []
        pf = []
        
        for i in range(1, len(data)):
            # Initialize variables for this iteration
            no_of_stocks = 0
            remain1 = 0
            tsl = 0
            ttp = 0
            max__drawdown2 = 0
            max__drawdown3 = 0
            price = 0
            
            # Create portfolio data structure (same as in notebook)
            if current_position == 1:
                slp = data['entry_price'].iloc[entry_index] - (self.atr_multiplier * data['ATR'].iloc[i])
                p2 = data['entry_price'].iloc[entry_index] + (self.atr_multiplier2 * data['ATR'].iloc[i])
            elif current_position == -1:
                slp = data['entry_price'].iloc[entry_index] + (self.atr_multiplier * data['ATR'].iloc[i])
                p2 = data['entry_price'].iloc[entry_index] - (self.atr_multiplier2 * data['ATR'].iloc[i])
            
            # Process signals exactly as in notebook
            if data['signals'].iloc[i] == 1 and current_position == 0:
                # Enter long position
                no_of_stocks = int(capital / data['Close'].iloc[i])
                num_stocks.append(no_of_stocks)
                price = data['Close'].iloc[i]
                take_profit_p = data['Close'].iloc[i] + (self.take_profit2/100) * data['Close'].iloc[i]
                stop_loss_p = data['Close'].iloc[i] - (self.stop_loss2/100) * data['Close'].iloc[i]
                remain1 = capital - no_of_stocks * price
                remains.append(remain1)
                
                current_position = 1
                entry_index = i
                
                # Update DataFrame
                data.loc[i, 'Position'] = 1
                data.loc[i, 'entry_price'] = price
                data.loc[i, 'stoploss_price'] = price - (self.atr_multiplier * data['ATR'].iloc[i])
                data.loc[i, 'trailing_sl'] = data.loc[i, 'stoploss_price']
                data.loc[i, 'trailing_tp'] = price + (self.take_profit/100 * price)
                data.loc[i, 'trade_type'] = TradeType.LONG.value
                
                # Calculate position size (exact notebook logic)
                risk_amount = 1000000 * 0.01  # 1% risk
                risk_per_unit = abs(price - data.loc[i, 'stoploss_price'])
                if risk_per_unit > 0:
                    data.loc[i, 'position_size'] = risk_amount / risk_per_unit
                
            # Long position with risk management exit conditions
            elif (current_position == 1 and 
                  ((abs((data['Low'].iloc[i] - data['Close'].iloc[i-1])) / data['Low'].iloc[i]) * 100 >= self.low_high_change_limit or
                   ((data['Close'].iloc[i-1] - data['Close'].iloc[i]) / data['Close'].iloc[i]) * 100 >= self.daily_change_limit or
                   (data['Close'].iloc[i] >= p2) or
                   ((data['Close'].iloc[i] - data['Low'].iloc[i]) / data['Close'].iloc[i]) * 100 >= self.change_limit or
                   (max__drawdown3 >= self.max_drawdown_limit2) or
                   (max__drawdown2 >= self.max_drawdown_limit) or
                   ((data['Close'].iloc[i] * no_of_stocks) + remain1 <= tsl) or
                   (data['Close'].iloc[i] <= slp) or
                   (data['Close'].iloc[i] >= take_profit_p) or
                   (data['Close'].iloc[i] <= stop_loss_p))):
                
                # Exit long position
                current_position = 0
                data.loc[i, 'Position'] = 0
                data.loc[i, 'trade_type'] = TradeType.CLOSE.value
                data.loc[i, 'trailing_sl'] = 0.0
                data.loc[i, 'trailing_tp'] = 0.0
                data.loc[i, 'position_size'] = 0.0
                
                # Update capital
                capital = (data['Close'].iloc[i] * no_of_stocks) + remain1
                data['signals'].iloc[i] = -1  # Signal to close long
                
            # Enter short position
            elif data['signals'].iloc[i] == -1 and current_position == 0:
                # Enter short position
                no_of_stocks = int(capital / data['Close'].iloc[i])
                num_stocks.append(no_of_stocks)
                price = data['Close'].iloc[i]
                take_profit_p = data['Close'].iloc[i] - (self.take_profit2/100) * data['Close'].iloc[i]
                stop_loss_p = data['Close'].iloc[i] + (self.stop_loss2/100) * data['Close'].iloc[i]
                remain1 = capital - no_of_stocks * price
                remains.append(0)
                
                current_position = -1
                entry_index = i
                
                # Update DataFrame
                data.loc[i, 'Position'] = -1
                data.loc[i, 'entry_price'] = price
                data.loc[i, 'stoploss_price'] = price + (self.atr_multiplier * data['ATR'].iloc[i])
                data.loc[i, 'trailing_sl'] = data.loc[i, 'stoploss_price']
                data.loc[i, 'trailing_tp'] = price - (self.take_profit/100 * price)
                data.loc[i, 'trade_type'] = TradeType.SHORT.value
                
                # Calculate position size
                risk_amount = 1000000 * 0.01  # 1% risk
                risk_per_unit = abs(price - data.loc[i, 'stoploss_price'])
                if risk_per_unit > 0:
                    data.loc[i, 'position_size'] = risk_amount / risk_per_unit
                
            # Short position with risk management exit conditions
            elif (current_position == -1 and 
                  ((abs((data['High'].iloc[i] - data['Close'].iloc[i-1])) / data['High'].iloc[i]) * 100 >= self.low_high_change_limit or
                   ((data['Close'].iloc[i] - data['Close'].iloc[i-1]) / data['Close'].iloc[i]) * 100 >= self.daily_change_limit or
                   (data['Close'].iloc[i] <= p2) or
                   ((data['High'].iloc[i] - data['Close'].iloc[i]) / data['Close'].iloc[i]) * 100 >= self.change_limit or
                   (capital + (no_of_stocks * (price - data['Close'].iloc[i])) >= ttp) or
                   (data['Close'].iloc[i] >= slp) or
                   (max__drawdown3 >= self.max_drawdown_limit2) or
                   (max__drawdown2 >= self.max_drawdown_limit) or
                   (data['Close'].iloc[i] <= take_profit_p) or
                   (data['Close'].iloc[i] >= stop_loss_p))):
                
                # Exit short position  
                current_position = 0
                data.loc[i, 'Position'] = 0
                data.loc[i, 'trade_type'] = TradeType.CLOSE.value
                data.loc[i, 'trailing_sl'] = 0.0
                data.loc[i, 'trailing_tp'] = 0.0
                data.loc[i, 'position_size'] = 0.0
                
                # Update capital
                capital = capital + ((price - data['Close'].iloc[i]) * no_of_stocks)
                data['signals'].iloc[i] = 1  # Signal to close short
                
            # Long position with sell signal
            elif data['signals'].iloc[i] == -1 and current_position == 1:
                # Exit long and enter short
                current_position = 0
                data.loc[i, 'Position'] = 0
                data.loc[i, 'trade_type'] = TradeType.CLOSE.value
                
                # Update capital
                capital = (data['Close'].iloc[i] * no_of_stocks) + remain1
                
            # Short position with buy signal
            elif data['signals'].iloc[i] == 1 and current_position == -1:
                # Exit short and enter long
                current_position = 0
                data.loc[i, 'Position'] = 0
                data.loc[i, 'trade_type'] = TradeType.CLOSE.value
                
                # Update capital
                capital = capital + ((price - data['Close'].iloc[i]) * no_of_stocks)
                
            # Cancel redundant signals (same direction as current position)
            elif current_position == 1 and data['signals'].iloc[i] == 1:
                data['signals'].iloc[i] = 0
            elif current_position == -1 and data['signals'].iloc[i] == -1:
                data['signals'].iloc[i] = 0
                
            # Update drawdown - exact notebook logic
            if i > 0:
                portfolio_value = data.loc[i-1, 'position_size'] * data['Close'].iloc[i]
                
                # Calculate max drawdown when in a position
                if current_position != 0:
                    # Update trailing stop loss/take profit logic for drawdown calculation
                    if current_position == 1:
                        # For long positions
                        if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
                            new_stop = data['Close'].iloc[i] - (self.atr_multiplier * data['ATR'].iloc[i])
                            if new_stop > data.loc[i-1, 'trailing_sl']:
                                data.loc[i, 'trailing_sl'] = new_stop
                            else:
                                data.loc[i, 'trailing_sl'] = data.loc[i-1, 'trailing_sl']
                        else:
                            data.loc[i, 'trailing_sl'] = data.loc[i-1, 'trailing_sl']
                            
                        # For trailing take profit calculation
                        data.loc[i, 'trailing_tp'] = data.loc[i-1, 'trailing_tp']
                    
                    elif current_position == -1:
                        # For short positions
                        if data['Close'].iloc[i] < data['Close'].iloc[i-1]:
                            new_stop = data['Close'].iloc[i] + (self.atr_multiplier * data['ATR'].iloc[i])
                            if new_stop < data.loc[i-1, 'trailing_sl']:
                                data.loc[i, 'trailing_sl'] = new_stop
                            else:
                                data.loc[i, 'trailing_sl'] = data.loc[i-1, 'trailing_sl']
                        else:
                            data.loc[i, 'trailing_sl'] = data.loc[i-1, 'trailing_sl']
                            
                        # For trailing take profit calculation
                        data.loc[i, 'trailing_tp'] = data.loc[i-1, 'trailing_tp']
                else:
                    # Clear trailing values when not in a position
                    data.loc[i, 'trailing_sl'] = 0.0
                    data.loc[i, 'trailing_tp'] = 0.0
                
            # Forward fill positions
            if i > 0 and data.loc[i, 'Position'] == 0 and data.loc[i, 'trade_type'] == TradeType.HOLD.value:
                data.loc[i, 'Position'] = data.loc[i-1, 'Position']
                if data.loc[i, 'Position'] != 0:
                    # Copy forward entry and stop loss prices
                    data.loc[i, 'entry_price'] = data.loc[i-1, 'entry_price']
                    data.loc[i, 'stoploss_price'] = data.loc[i-1, 'stoploss_price']
                    data.loc[i, 'position_size'] = data.loc[i-1, 'position_size']
                    
                    # Update trailing values
                    if current_position == 1:  # Long position
                        if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
                            new_stop = data['Close'].iloc[i] - (self.atr_multiplier * data['ATR'].iloc[i])
                            if new_stop > data.loc[i-1, 'trailing_sl']:
                                data.loc[i, 'trailing_sl'] = new_stop
                            else:
                                data.loc[i, 'trailing_sl'] = data.loc[i-1, 'trailing_sl']
                        else:
                            data.loc[i, 'trailing_sl'] = data.loc[i-1, 'trailing_sl']
                        
                        data.loc[i, 'trailing_tp'] = data.loc[i-1, 'trailing_tp']
                    
                    elif current_position == -1:  # Short position
                        if data['Close'].iloc[i] < data['Close'].iloc[i-1]:
                            new_stop = data['Close'].iloc[i] + (self.atr_multiplier * data['ATR'].iloc[i])
                            if new_stop < data.loc[i-1, 'trailing_sl']:
                                data.loc[i, 'trailing_sl'] = new_stop
                            else:
                                data.loc[i, 'trailing_sl'] = data.loc[i-1, 'trailing_sl']
                        else:
                            data.loc[i, 'trailing_sl'] = data.loc[i-1, 'trailing_sl']
                        
                        data.loc[i, 'trailing_tp'] = data.loc[i-1, 'trailing_tp']
        
        # Final clean-up of any NaN values
        for col in ['Position', 'entry_price', 'stoploss_price', 'trailing_sl', 'trailing_tp', 'position_size']:
            data[col] = data[col].fillna(0)
        
        return data
    
    def backtest(self, data, initial_balance=1000000.0):
        """
        Run a backtest of the strategy with risk management using the exact framework from the notebook.
        
        Parameters:
            data (pd.DataFrame): Price data with signals already generated
            initial_balance (float): Initial capital for backtesting
            
        Returns:
            tuple: (trade_df, portfolio_df, buy_signals, sell_signals)
        """
        # Create a copy of the data to avoid modifying the original
        data9 = data.copy()
        
        # Create portfolio tracking dataframe
        dd1 = pd.DataFrame()
        dd1['No_of_Stocks'] = 0
        dd1['Portfolio_Value'] = 0
        dd1['Profit_From_Initial_Capital'] = 0
        dd1['Current_Position'] = 0
        
        # Track positions
        num_buys = 0  # Total no. of buy signals
        num_sells = 0  # Total no. of sell signals
        buys = []  # List containing index of each buy trade
        sells = []  # List containing index of each sell trade
        trades = []  # List containing index of all trades
        
        # Count buy and sell signals
        for i in range(len(data9)):
            if data9['signals'].iloc[i] == 1:
                num_buys += 1
                buys.append(i)
                trades.append(i)
            elif data9['signals'].iloc[i] == -1:
                num_sells += 1
                sells.append(i)
                trades.append(i)
                
        # Ensure we have equal number of buy and sell signals (or handle the last trade)
        if num_buys != num_sells:
            # Changing the signal of the last trade to 0 (no position)
            data9['signals'].iloc[trades[num_buys + num_sells - 1]] = 0
            trades.pop()
            if num_buys > num_sells:
                buys.pop()
            else:
                sells.pop()
            num_closetrades = max(num_buys, num_sells) - 1  # Total no. of close trades
        else:
            num_closetrades = num_buys
        
        # Set initial balance and capital
        balance = initial_balance
        capital = initial_balance
        
        # Risk Management Parameters (same as notebook)
        take_profit = 10  # Percentage for trailing take profit
        take_profit2 = 10  # Percentage for normal take profit
        multiplier = 1  # Multiplier for Average True Range Stop Loss
        maxDrawdownLimit = 15  # Drawdown Limit to Exit a Trade (drawdown calculated for entire period)
        percent = 7  # Percentage limit of change in (close-low) for long and (high-close) for short trades
        percent2 = 7  # Percentage limit of daily change in close price
        
        # Risk Management Measures not in use (but kept for consistency with notebook)
        stop_loss = 100  # Percentage for trailing stop loss
        stop_loss2 = 100  # Percentage for normal stop loss
        multiplier2 = 10  # Multiplier for Average True Range Take Profit
        maxDrawdownLimit2 = 100  # Drawdown Limit to Exit a Trade (drawdown within a trade)
        percent3 = 100  # Percentage limit of change in low of current day to close of previous day
        
        # Initialize tracking variables
        no_of_stocks = 0
        position = 0
        num_stocks = []  # List of no. of stocks hold in each trade
        returns = []  # List containing return from each trade
        capitall = []  # List containing capital after each trade
        remains = []  # List containing not invested capital for each trade
        remainss = []  # List containing not invested capital for each trade
        max__drawdown = []  # List containing Max Drawdown for each trade
        max__dip = []  # List containing Max Dip for each trade
        pv = []
        j = 0
        l = 0
        pf = []
        trades = []
        rfr1 = []
        buy_signals = []
        sell_signals = []
        tsll = []
        ttpl = []
        drawdown2 = []
        mx = 1000000
        dd2 = []
        
        # Create tradewise dataframe to track all trades
        twd1 = pd.DataFrame(columns=['Entry Index', 'Exit Index', 'Trade Duration', 'Entry Price', 
                                    'Exit Price', 'Returns for Trade in %', 'Type of Trade',
                                    'Trade Close By', 'No. of Stocks Traded', 'Profit/Loss', 'Capital'])
        
        # Main backtesting loop
        for i in range(len(data9)):
            if capital > 0:
                # Calculate stop loss and take profit prices for current position
                if position == 1:
                    slp = twd1['Entry Price'].iloc[j] - (multiplier * data9['ATR'].iloc[i])  # Stop Loss price for Long Trades using ATR
                    p2 = twd1['Entry Price'].iloc[j] + (multiplier2 * data9['ATR'].iloc[i])  # Take Profit price for Long Trades using ATR
                elif position == -1:
                    slp = twd1['Entry Price'].iloc[j] + (multiplier * data9['ATR'].iloc[i])  # Stop Loss price for Short Trades using ATR
                    p2 = twd1['Entry Price'].iloc[j] - (multiplier2 * data9['ATR'].iloc[i])  # Take Profit price for Short Trades using ATR
                
                # Taking a Long position
                if data9['signals'].iloc[i] == 1 and position == 0:
                    no_of_stocks = int(capital / data9['Close'].iloc[i])
                    num_stocks.append(no_of_stocks)
                    price = data9['Close'].iloc[i]
                    take_profit_p = data9['Close'].iloc[i] + (take_profit2 / 100) * data9['Close'].iloc[i]
                    stop_loss_p = data9['Close'].iloc[i] - (stop_loss2 / 100) * data9['Close'].iloc[i]
                    remain1 = capital - no_of_stocks * price
                    remains.append(remain1)
                    
                    # Filling different columns of the DataFrame twd1
                    new_row = {'Entry Index': i, 'Exit Index': 0, 'Trade Duration': 0, 'Entry Price': price,
                              'Returns for Trade in %': 0, 'Type of Trade': "long",
                              'Trade Close By': "none", 'Max Drawdown for Trade': 0, 'Max Dip for Trade': 0,
                              'No. of Stocks Traded': no_of_stocks, 'Profit/Loss': 0, 'Capital': 0}
                    twd1.loc[len(twd1)] = new_row
                    position = 1
                    trades.append(i)
                
                # Long position with stop loss or take profit condition (exact same as notebook)
                elif (position == 1 and (
                        ((abs((data9['Low'].iloc[i] - data9['Close'].iloc[i-1])) / data9['Low'].iloc[i]) * 100) >= percent3 or
                        (((data9['Close'].iloc[i-1] - data9['Close'].iloc[i]) / data9['Close'].iloc[i]) * 100) >= percent2 or
                        (data9['Close'].iloc[i] >= p2) or
                        (((data9['Close'].iloc[i] - data9['Low'].iloc[i]) / data9['Close'].iloc[i]) * 100) >= percent or
                        (max__drawdown3 >= maxDrawdownLimit2) or
                        (max__drawdown2 >= maxDrawdownLimit) or
                        (((data9['Close'].iloc[i] * no_of_stocks) + remain1) <= tsl) or
                        (data9['Close'].iloc[i] <= slp) or
                        (data9['Close'].iloc[i] >= take_profit_p) or
                        (data9['Close'].iloc[i] <= stop_loss_p))):
                    
                    # Determine exit reason
                    if (((data9['Close'].iloc[i] * no_of_stocks) + remain1) <= tsl):
                        twd1['Trade Close By'].iloc[j] = "trailing_stop_loss"
                    elif (data9['Close'].iloc[i] >= take_profit_p):
                        twd1['Trade Close By'].iloc[j] = "take_profit"
                    elif max__drawdown2 >= maxDrawdownLimit:
                        twd1['Trade Close By'].iloc[j] = "Max Drawdown Limit"
                    elif max__drawdown3 >= maxDrawdownLimit2:
                        twd1['Trade Close By'].iloc[j] = "Max Drawdown Limit2"
                    elif data9['Close'].iloc[i] <= slp:
                        twd1['Trade Close By'].iloc[j] = "ATR_stop_loss1"
                    elif data9['Close'].iloc[i] <= stop_loss_p:
                        twd1['Trade Close By'].iloc[j] = "stop_loss2"
                    elif ((data9['Close'].iloc[i] - data9['Low'].iloc[i]) / data9['Close'].iloc[i]) * 100 >= percent:
                        twd1['Trade Close By'].iloc[j] = "ID"
                    elif ((data9['Close'].iloc[i-1] - data9['Close'].iloc[i]) / data9['Close'].iloc[i]) * 100 >= percent2:
                        twd1['Trade Close By'].iloc[j] = "nextday"
                    elif ((abs((data9['Low'].iloc[i] - data9['Close'].iloc[i-1])) / data9['Low'].iloc[i]) * 100) >= percent3:
                        twd1['Trade Close By'].iloc[j] = "low"
                    elif data9['Close'].iloc[i] >= p2:
                        twd1['Trade Close By'].iloc[j] = "ATR_TP"
                    
                    # Calculate returns and update position
                    returns1 = (((data9['Close'].iloc[i] - price) / price) * 100)
                    capital = (data9['Close'].iloc[i] * no_of_stocks) + remain1
                    net = twd1['No. of Stocks Traded'].iloc[j] * ((data9['Close'].iloc[i] - twd1['Entry Price'].iloc[j]))
                    twd1['Exit Index'].iloc[j] = i
                    twd1['Exit Price'].iloc[j] = data9['Close'].iloc[i]
                    twd1['Trade Duration'].iloc[j] = twd1['Exit Index'].iloc[j] - twd1['Entry Index'].iloc[j]
                    twd1['Returns for Trade in %'].iloc[j] = returns1
                    twd1['Profit/Loss'].iloc[j] = net
                    twd1['Capital'].iloc[j] = capital
                    data9['signals'].iloc[i] = -1
                    j = j + 1
                    position = 0
                    no_of_stocks = 0
                    trades.append(i)
                
                # Taking a Short position
                elif data9['signals'].iloc[i] == -1 and position == 0:
                    no_of_stocks = int(capital / data9['Close'].iloc[i])
                    num_stocks.append(no_of_stocks)
                    price = data9['Close'].iloc[i]
                    take_profit_p = data9['Close'].iloc[i] - (take_profit2 / 100) * data9['Close'].iloc[i]
                    stop_loss_p = data9['Close'].iloc[i] + (stop_loss2 / 100) * data9['Close'].iloc[i]
                    remain1 = capital - no_of_stocks * price
                    remains.append(0)
                    
                    # Filling different columns of the DataFrame twd1
                    new_row = {'Entry Index': i, 'Exit Index': 0, 'Trade Duration': 0, 'Entry Price': price,
                              'Returns for Trade in %': 0, 'Type of Trade': "short",
                              'Trade Close By': "none", 'Max Drawdown for Trade': 0, 'Max Dip for Trade': 0,
                              'No. of Stocks Traded': no_of_stocks, 'Profit/Loss': 0, 'Capital': 0}
                    twd1.loc[len(twd1)] = new_row
                    position = -1
                    trades.append(i)
                
                # Short position with stop loss or take profit condition (exact same as notebook)
                elif (position == -1 and (
                        ((abs((data9['High'].iloc[i] - data9['Close'].iloc[i-1])) / data9['High'].iloc[i]) * 100) >= percent3 or
                        (((data9['Close'].iloc[i] - data9['Close'].iloc[i-1]) / data9['Close'].iloc[i]) * 100) >= percent2 or
                        (data9['Close'].iloc[i] <= p2) or
                        (((data9['High'].iloc[i] - data9['Close'].iloc[i]) / data9['Close'].iloc[i]) * 100) >= percent or
                        ((capital + (no_of_stocks * (price - data9['Close'].iloc[i]))) >= ttp) or
                        (data9['Close'].iloc[i] >= slp) or
                        (max__drawdown3 >= maxDrawdownLimit2) or
                        (max__drawdown2 >= maxDrawdownLimit) or
                        (data9['Close'].iloc[i] <= take_profit_p) or
                        (data9['Close'].iloc[i] >= stop_loss_p))):
                    
                    # Determine exit reason
                    if ((capital + (no_of_stocks * (price - data9['Close'].iloc[i]))) >= ttp):
                        twd1['Trade Close By'].iloc[j] = "trailing_take_profit"
                    elif (data9['Close'].iloc[i] <= take_profit_p):
                        twd1['Trade Close By'].iloc[j] = "take_profit"
                    elif max__drawdown2 >= maxDrawdownLimit:
                        twd1['Trade Close By'].iloc[j] = "Max Drawdown Limit"
                    elif max__drawdown3 >= maxDrawdownLimit2:
                        twd1['Trade Close By'].iloc[j] = "Max Drawdown Limit2"
                    elif data9['Close'].iloc[i] >= slp:
                        twd1['Trade Close By'].iloc[j] = "ATR_stop_loss2"
                    elif data9['Close'].iloc[i] >= stop_loss_p:
                        twd1['Trade Close By'].iloc[j] = "stop_loss2"
                    elif ((data9['High'].iloc[i] - data9['Close'].iloc[i]) / data9['Close'].iloc[i]) * 100 >= percent:
                        twd1['Trade Close By'].iloc[j] = "ID"
                    elif ((data9['Close'].iloc[i] - data9['Close'].iloc[i-1]) / data9['Close'].iloc[i]) * 100 >= percent2:
                        twd1['Trade Close By'].iloc[j] = "nextday"
                    elif ((abs((data9['High'].iloc[i] - data9['Close'].iloc[i-1])) / data9['High'].iloc[i]) * 100) >= percent3:
                        twd1['Trade Close By'].iloc[j] = "high"
                    elif data9['Close'].iloc[i] <= p2:
                        twd1['Trade Close By'].iloc[j] = "ATR_TP"
                    
                    # Calculate returns and update position
                    returns1 = (((price - data9['Close'].iloc[i]) / price) * 100)
                    capital = capital + ((price - data9['Close'].iloc[i]) * no_of_stocks)
                    net = twd1['No. of Stocks Traded'].iloc[j] * ((twd1['Entry Price'].iloc[j] - data9['Close'].iloc[i]))
                    twd1['Exit Index'].iloc[j] = i
                    twd1['Exit Price'].iloc[j] = data9['Close'].iloc[i]
                    twd1['Trade Duration'].iloc[j] = twd1['Exit Index'].iloc[j] - twd1['Entry Index'].iloc[j]
                    twd1['Returns for Trade in %'].iloc[j] = returns1
                    twd1['Profit/Loss'].iloc[j] = net
                    twd1['Capital'].iloc[j] = capital
                    data9['signals'].iloc[i] = 1
                    j = j + 1
                    position = 0
                    no_of_stocks = 0
                    trades.append(i)
                
                # Long position with sell signal
                elif data9['signals'].iloc[i] == -1 and position == 1:
                    returns1 = (((data9['Close'].iloc[i] - price) / price) * 100)
                    capital = (data9['Close'].iloc[i] * no_of_stocks) + remain1
                    net = twd1['No. of Stocks Traded'].iloc[j] * ((data9['Close'].iloc[i] - twd1['Entry Price'].iloc[j]))
                    twd1['Exit Index'].iloc[j] = i
                    twd1['Exit Price'].iloc[j] = data9['Close'].iloc[i]
                    twd1['Trade Duration'].iloc[j] = twd1['Exit Index'].iloc[j] - twd1['Entry Index'].iloc[j]
                    twd1['Returns for Trade in %'].iloc[j] = returns1
                    twd1['Trade Close By'].iloc[j] = "signal"
                    twd1['Profit/Loss'].iloc[j] = net
                    twd1['Capital'].iloc[j] = capital
                    data9['signals'].iloc[i] = -1
                    j = j + 1
                    position = 0
                    no_of_stocks = 0
                    trades.append(i)
                
                # Short position with buy signal condition
                elif data9['signals'].iloc[i] == 1 and position == -1:
                    returns1 = (((price - data9['Close'].iloc[i]) / price) * 100)
                    capital = capital + ((price - data9['Close'].iloc[i]) * no_of_stocks)
                    net = twd1['No. of Stocks Traded'].iloc[j] * ((twd1['Entry Price'].iloc[j] - data9['Close'].iloc[i]))
                    twd1['Exit Index'].iloc[j] = i
                    twd1['Exit Price'].iloc[j] = data9['Close'].iloc[i]
                    twd1['Trade Duration'].iloc[j] = twd1['Exit Index'].iloc[j] - twd1['Entry Index'].iloc[j]
                    twd1['Returns for Trade in %'].iloc[j] = returns1
                    twd1['Trade Close By'].iloc[j] = "signal"
                    twd1['Profit/Loss'].iloc[j] = net
                    twd1['Capital'].iloc[j] = capital
                    data9['signals'].iloc[i] = 1
                    j = j + 1
                    position = 0
                    no_of_stocks = 0
                    trades.append(i)
                
                # Handle redundant signals
                elif position == 1 and data9['signals'].iloc[i] == 1:
                    data9['signals'].iloc[i] = 0
                elif position == -1 and data9['signals'].iloc[i] == -1:
                    data9['signals'].iloc[i] = 0
                
                # Update portfolio values
                dd1.loc[i, 'No_of_Stocks'] = no_of_stocks
                if no_of_stocks != 0:
                    if position == 1:
                        dd1.loc[i, 'Portfolio_Value'] = (no_of_stocks * data9['Close'].iloc[i]) + remain1
                    else:
                        dd1.loc[i, 'Portfolio_Value'] = capital + (no_of_stocks * (price - data9['Close'].iloc[i]))
                else:
                    dd1.loc[i, 'Portfolio_Value'] = capital
                
                dd1.loc[i, 'Profit_From_Initial_Capital'] = ((dd1['Portfolio_Value'].iloc[i] - balance) / balance) * 100
                
                # Calculate maximum drawdown
                mx = max(dd1['Portfolio_Value'].iloc[:i+1])
                if dd1['Portfolio_Value'].iloc[i] > mx:
                    mx = dd1['Portfolio_Value'].iloc[i]
                
                dd2.append(((mx - dd1['Portfolio_Value'].iloc[i]) / dd1['Portfolio_Value'].iloc[i]) * 100)
                max__drawdown2 = dd2[i]
                
                # Calculate drawdown per position
                if position == 1:
                    max2 = dd1['Portfolio_Value'].iloc[twd1['Entry Index'].iloc[j]]
                    for k in range(twd1['Entry Index'].iloc[j], i+1):
                        pf.append(dd1['Portfolio_Value'].iloc[k])
                        maxx = max(pf)
                        portfolio2 = dd1['Portfolio_Value'].iloc[k]
                        if portfolio2 > max2:
                            max2 = portfolio2
                        drawdown2.append(((max2 - portfolio2) / portfolio2) * 100)
                    max__drawdown3 = max(drawdown2)
                    tsl = (1 - (stop_loss / 100)) * maxx
                    tsll.append(tsl)
                    pf.clear()
                    drawdown2.clear()
                
                if position == -1:
                    max2 = dd1['Portfolio_Value'].iloc[twd1['Entry Index'].iloc[j]]
                    for k in range(twd1['Entry Index'].iloc[j], i+1):
                        pf.append(dd1['Portfolio_Value'].iloc[k])
                        minn = min(pf)
                        portfolio2 = dd1['Portfolio_Value'].iloc[k]
                        if portfolio2 > max2:
                            max2 = portfolio2
                        drawdown2.append(((max2 - portfolio2) / portfolio2) * 100)
                    max__drawdown3 = max(drawdown2)
                    ttp = (1 + (take_profit / 100)) * minn
                    ttpl.append(ttp)
                    pf.clear()
                    drawdown2.clear()
                
                # Manage tracking variables for stop/take levels
                if position == 0 or position == -1:
                    tsll.append(dd1['Portfolio_Value'].iloc[i])
                if position == 0 or position == 1:
                    ttpl.append(dd1['Portfolio_Value'].iloc[i])
        
        # Handle any unmatched trades
        if len(twd1) > 0 and twd1['Exit Index'].iloc[len(twd1)-1] == 0:
            data9['signals'].iloc[twd1['Entry Index'].iloc[len(twd1)-1]] = 0
            twd1 = twd1.drop(len(twd1)-1, axis=0)
            trades.pop()
        
        # Collect final buy/sell signals for plotting
        for i in range(len(data9)):
            if data9['signals'].iloc[i] == 1:
                buy_signals.append(i)
            elif data9['signals'].iloc[i] == -1:
                sell_signals.append(i)
        
        # Calculate drawdown for each trade
        dd1['Current Position'] = 0
        dd1['drawdown'] = 0
        a = 0
        for i in range(0, ((int(len(trades)/2)))*2, 2):
            drawdown = []
            index1 = trades[i]
            index2 = trades[i+1]
            stocks = num_stocks[a]
            remain = remains[a]
            max1 = dd1['Portfolio_Value'].iloc[index1]
            min1 = dd1['Portfolio_Value'].iloc[index1]
            for j in range(index1, index2+1):
                portfolio = dd1['Portfolio_Value'].iloc[j]
                if portfolio > max1:
                    max1 = portfolio
                if portfolio < min1:
                    min1 = portfolio
                drawdown.append(((max1 - portfolio) / portfolio) * 100)
                dd1.loc[j, 'drawdown'] = ((max1 - portfolio) / portfolio) * 100
            
            max__drawdown.append(max(drawdown))
            max__dip.append(((dd1['Portfolio_Value'].iloc[index1] - min1) / (dd1['Portfolio_Value'].iloc[index1])) * 100)
            a = a + 1
        
        # Add drawdown metrics to tradewise dataframe
        if len(max__drawdown) > 0 and len(twd1) > 0:
            if len(max__drawdown) == len(twd1):
                twd1['Max Drawdown for Trade'] = max__drawdown
                twd1['Max Dip for Trade'] = max__dip
            
        dd1['drawdown2'] = dd2
        
        return twd1, dd1, buy_signals, sell_signals

    def run_backtest(self, df, equity=1000000.0, start_date='2020-01-01', end_date='2023-12-31'):
        """
        Run a complete backtest of the strategy with performance metrics.
        
        Parameters:
            df (pd.DataFrame): Price data containing OHLCV columns
            equity (float): Initial capital for backtesting
            start_date (str): Start date for backtest in YYYY-MM-DD format (default: 2020-01-01)
            end_date (str): End date for backtest in YYYY-MM-DD format (default: 2023-12-31)
            
        Returns:
            tuple: (strategy_data, trade_df, portfolio_df, performance_metrics)
        """
        # Make a copy of the data
        df_copy = df.copy()
        
        # Ensure column names are consistent - exact match with notebook expectations
        for old, new in [
            ('open', 'Open'), ('high', 'High'), ('low', 'Low'), ('close', 'Close'), ('volume', 'Volume')
        ]:
            if old in df_copy.columns and new not in df_copy.columns:
                df_copy[new] = df_copy[old]
            elif new in df_copy.columns and old not in df_copy.columns:
                df_copy[old] = df_copy[new]
                
        # Ensure we have a datetime index or column
        if 'date' in df_copy.columns:
            df_copy['date'] = pd.to_datetime(df_copy['date'])
            date_col = 'date'
        elif 'timestamp' in df_copy.columns:
            df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
            date_col = 'timestamp'
        elif 'time' in df_copy.columns:
            df_copy['time'] = pd.to_datetime(df_copy['time'])
            date_col = 'time'
        else:
            # Try to convert index to datetime if not already a column
            try:
                df_copy.index = pd.to_datetime(df_copy.index)
                date_col = None  # Using index
            except:
                print("Warning: No date column found. Using all data without date filtering.")
                date_col = None

        # Filter data by date range if date column exists - exactly like notebook
        if date_col:
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
                
            if date_col is None:  # Date is in index
                filtered_data = df_copy[(df_copy.index >= start_date) & (df_copy.index <= end_date)]
            else:  # Date is in column
                filtered_data = df_copy[(df_copy[date_col] >= start_date) & (df_copy[date_col] <= end_date)]
                
            print(f"Filtered data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            print(f"Original data points: {len(df_copy)}, Filtered data points: {len(filtered_data)}")
        else:
            filtered_data = df_copy
            
        # Ensure lowercase volume column for indicator calculations - exact match with notebook
        if 'Volume' in filtered_data.columns and 'volume' not in filtered_data.columns:
            filtered_data['volume'] = filtered_data['Volume']
            
        # Important: Reset index after filtering to avoid KeyError: 0
        filtered_data = filtered_data.reset_index(drop=True)
        
        # Set strategy parameters to match the notebook exactly
        self.take_profit = 10
        self.take_profit2 = 10
        self.stop_loss = 100
        self.stop_loss2 = 100
        self.atr_multiplier = 1.0
        self.atr_multiplier2 = 10.0
        self.max_drawdown_limit = 15
        self.max_drawdown_limit2 = 100
        self.change_limit = 7
        self.daily_change_limit = 7
        self.low_high_change_limit = 100
        self.rsi_period = 14
        
        # Run the main strategy to generate signals
        data = self.run(filtered_data, equity)
        
        # Run the backtesting framework from the notebook
        trade_df, portfolio_df, buy_signals, sell_signals = self.backtest(data, equity)
        
        # Calculate performance metrics
        performance = {}
        
        if len(trade_df) > 0:
            # Benchmark return
            benchmark_return = ((filtered_data['close'].iloc[-1] - filtered_data['close'].iloc[0]) / filtered_data['close'].iloc[0]) * 100
            performance['benchmark_return'] = benchmark_return
            
            # Strategy return
            strategy_return = portfolio_df['Profit_From_Initial_Capital'].iloc[-1]
            performance['strategy_return'] = strategy_return
            
            # Number of trades
            performance['total_trades'] = len(trade_df)
            
            # Win rate
            winning_trades = len(trade_df[trade_df['Returns for Trade in %'] > 0])
            performance['winning_trades'] = winning_trades
            performance['win_rate'] = (winning_trades / len(trade_df)) * 100
            
            # Max drawdown
            if 'Max Drawdown for Trade' in trade_df.columns:
                performance['max_drawdown'] = trade_df['Max Drawdown for Trade'].max()
                performance['avg_drawdown'] = trade_df['Max Drawdown for Trade'].mean()
            
            # Trade types
            long_trades = len(trade_df[trade_df['Type of Trade'] == 'long'])
            short_trades = len(trade_df[trade_df['Type of Trade'] == 'short'])
            performance['long_trades'] = long_trades
            performance['short_trades'] = short_trades
            
            # Average holding time
            performance['avg_holding_time'] = trade_df['Trade Duration'].mean()
            
            # Profit/Loss
            performance['total_profit'] = trade_df['Profit/Loss'].sum()
            
            # Calculate Sharpe Ratio
            # 1. Get trade returns
            returns = trade_df['Returns for Trade in %'].values
            
            # 2. Calculate mean and standard deviation
            mean_returns = np.mean(returns) if len(returns) > 0 else 0
            std_returns = np.std(returns) if len(returns) > 0 else 1  # Avoid division by zero
            
            # 3. Estimate risk-free rate (we can use a standard value like 2% annualized)
            risk_free_rate = 2.0  # Annual percentage
            
            # 4. Normalize based on average holding period
            avg_holding_period = performance['avg_holding_time'] if 'avg_holding_time' in performance else 1
            trades_per_year = 365 / max(1, avg_holding_period)  # Estimated number of trades per year
            
            # 5. Calculate annualized Sharpe ratio
            if std_returns > 0:
                sharpe_ratio = (mean_returns - (risk_free_rate / trades_per_year)) / std_returns * np.sqrt(trades_per_year)
            else:
                sharpe_ratio = 0
                
            performance['sharpe_ratio'] = sharpe_ratio
            
            # Print performance summary
            print(f"Performance metrics for {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}:")
            print("Benchmark Return: {:.2f}%".format(benchmark_return))
            print("Strategy Return: {:.2f}%".format(strategy_return))
            print("Total Trades: {}".format(len(trade_df)))
            print("Win Rate: {:.2f}%".format((winning_trades / len(trade_df)) * 100))
            if 'Max Drawdown for Trade' in trade_df.columns:
                print("Maximum Drawdown: {:.2f}%".format(trade_df['Max Drawdown for Trade'].max()))
            print("Sharpe Ratio: {:.2f}".format(sharpe_ratio))
            print("Long Trades: {}".format(long_trades))
            print("Short Trades: {}".format(short_trades))
        else:
            print(f"No trades were executed in the period from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        return data, trade_df, portfolio_df, performance


if __name__ == "__main__":
    # Create strategy instance
    strategy = Strategy()
    
    # Load data (adjust path as needed)
    try:
        data = pd.read_csv("/home/mayank/work_space/AIMS/BITS/zelta/model/cookin/BTCUSDT_historical_data_1d.csv")
        
        # Run backtest for 2020-2023 data
        result_data, trades_df, portfolio_df, performance = strategy.run_backtest(
            data, 
            equity=1000000.0, 
            start_date='2020-01-01', 
            end_date='2023-12-31'
        )
        
        # Save results (optional)
        result_data['signals'].to_csv("/home/mayank/work_space/AIMS/BITS/zelta/model/cookin/signals_2020_2023.csv", index=False)
        trades_df.to_csv("/home/mayank/work_space/AIMS/BITS/zelta/model/cookin/trades_2020_2023.csv", index=False)
        portfolio_df.to_csv("/home/mayank/work_space/AIMS/BITS/zelta/model/cookin/portfolio_2020_2023.csv", index=False)
        
        print("Backtest for 2020-2023 completed successfully!")
    except Exception as e:
        import traceback
        print(f"Error running backtest: {e}")
        traceback.print_exc()
