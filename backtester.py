import pandas as pd
import numpy as np
from datetime import datetime

class Backtester:
    def __init__(self, data=None, strategy=None, initial_capital=10000.0, transaction_cost=0.001):
        """
        Initialize the backtester.
        
        Args:
            data: DataFrame with price data
            strategy: Strategy object that generates signals
            initial_capital: Starting capital for the backtest
            transaction_cost: Cost of making a transaction (percentage)
        """
        self.data = data
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.results = None
        
    def run(self, signals_data=None):
        """
        Run backtest on the given signals data.
        
        Args:
            signals_data: DataFrame with 'close' prices and 'signal' column (1 for buy, -1 for sell, 0 for hold)
        
        Returns:
            DataFrame with backtest results
        """
        # Use provided signals_data or generate signals using the strategy
        if signals_data is None and self.data is not None and self.strategy is not None:
            signals_data = self.strategy.generate_signals(self.data)
        
        if signals_data is None:
            raise ValueError("No data provided for backtesting")
            
        # Create a copy of the signals data to avoid modifying the original
        results = signals_data.copy()
        
        # Add columns for tracking positions and portfolio
        results['position'] = 0.0
        results['cash'] = self.initial_capital
        results['holdings'] = 0.0
        results['portfolio'] = self.initial_capital
        
        # Track the current position
        position = 0.0
        cash = self.initial_capital
        
        # Loop through the data (excluding the first row)
        for i in range(1, len(results)):
            # Get the current price and signal - use iloc for positional indexing
            price = results['close'].iloc[i]
            signal = results['signal'].iloc[i]
            prev_signal = results['signal'].iloc[i-1]
            
            # Check for signal changes
            if signal != prev_signal:
                # If buying
                if signal == 1:
                    # Calculate how many units we can buy
                    units_to_buy = cash / price / (1 + self.transaction_cost)
                    cost = units_to_buy * price * (1 + self.transaction_cost)
                    
                    # Update position and cash
                    position += units_to_buy
                    cash -= cost
                
                # If selling
                elif signal == -1 and position > 0:
                    # Sell all current position
                    sell_value = position * price * (1 - self.transaction_cost)
                    
                    # Update position and cash
                    position = 0
                    cash += sell_value
            
            # Update the results
            results['position'].iloc[i] = position
            results['cash'].iloc[i] = cash
            results['holdings'].iloc[i] = position * price
            results['portfolio'].iloc[i] = cash + results['holdings'].iloc[i]
        
        self.results = results
        return results
    
    def calculate_metrics(self):
        """
        Calculate performance metrics from the backtest results.
        
        Returns:
            Dictionary with performance metrics
        """
        if self.results is None or len(self.results) == 0:
            return {"error": "No backtest results available"}
        
        # Calculate returns
        self.results['returns'] = self.results['portfolio'].pct_change()
        
        # Calculate performance metrics
        total_return = (self.results['portfolio'].iloc[-1] / self.initial_capital) - 1
        annual_return = (1 + total_return) ** (252 / len(self.results)) - 1
        sharpe_ratio = np.sqrt(252) * self.results['returns'].mean() / self.results['returns'].std()
        max_drawdown = (self.results['portfolio'] / self.results['portfolio'].cummax() - 1).min()
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
        
    def plot_results(self):
        """
        Plot the backtest results.
        
        Returns:
            Matplotlib figure
        """
        if self.results is None or len(self.results) == 0:
            print("No backtest results available")
            return None
        
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            
            # Plot portfolio value
            ax1.plot(self.results.index, self.results['portfolio'])
            ax1.set_title('Portfolio Value')
            ax1.set_ylabel('Value ($)')
            ax1.grid(True)
            
            # Plot buy/sell signals
            ax2.plot(self.results.index, self.results['close'])
            
            # Buy signals
            buy_signals = self.results[self.results['signal'] == 1]
            ax2.scatter(buy_signals.index, buy_signals['close'], marker='^', color='g', label='Buy')
            
            # Sell signals
            sell_signals = self.results[self.results['signal'] == -1]
            ax2.scatter(sell_signals.index, sell_signals['close'], marker='v', color='r', label='Sell')
            
            ax2.set_title('Price and Signals')
            ax2.set_ylabel('Price ($)')
            ax2.grid(True)
            ax2.legend()
            
            plt.tight_layout()
            return fig
        except ImportError:
            print("Matplotlib is required for plotting")
            return None
