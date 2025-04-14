import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class PerformanceAnalyzer:
    def __init__(self, equity_curve, trades, risk_free_rate=0.01):
        self.equity_curve = equity_curve
        self.trades = trades
        self.risk_free_rate = risk_free_rate
        
    def calculate_total_return(self):
        """Calculate total return of the strategy."""
        initial_equity = self.equity_curve.iloc[0]
        final_equity = self.equity_curve.iloc[-1]
        total_return = (final_equity - initial_equity) / initial_equity * 100
        return total_return
    
    def calculate_max_drawdown(self):
        """Calculate maximum drawdown and time to recover."""
        # Calculate drawdown series
        peak = self.equity_curve.cummax()
        drawdown = (self.equity_curve - peak) / peak * 100
        
        # Find maximum drawdown and its period
        max_dd = drawdown.min()
        max_dd_idx = drawdown.idxmin()
        
        # Find recovery time
        if max_dd_idx == self.equity_curve.index[-1] or all(drawdown[max_dd_idx:] < 0):
            # Not recovered yet
            recovery_time = pd.Timedelta(days=0)
            recovery_idx = None
        else:
            # Find the first index after max_dd where equity returns to the previous peak
            recovery_indices = self.equity_curve[max_dd_idx:][self.equity_curve[max_dd_idx:] >= peak[max_dd_idx]].index
            if len(recovery_indices) > 0:
                recovery_idx = recovery_indices[0]
                recovery_time = recovery_idx - max_dd_idx
            else:
                recovery_idx = None
                recovery_time = pd.Timedelta(days=0)
            
        return {
            'max_drawdown': max_dd,
            'max_drawdown_date': max_dd_idx,
            'recovery_date': recovery_idx,
            'recovery_time': recovery_time
        }
    
    def calculate_sharpe_ratio(self):
        """Calculate the Sharpe ratio."""
        # Calculate daily returns - using resample to avoid threading issues
        daily_returns = self.equity_curve.resample('D').last().pct_change().dropna()
        
        # Annualize returns and volatility (assuming 252 trading days per year)
        annual_return = daily_returns.mean() * 252
        annual_volatility = daily_returns.std() * np.sqrt(252)
        
        # Calculate Sharpe ratio
        if annual_volatility == 0:
            return 0  # Avoid division by zero
        
        sharpe_ratio = (annual_return - self.risk_free_rate) / annual_volatility
        
        return sharpe_ratio
    
    def generate_summary(self):
        """Generate a summary of performance metrics."""
        total_return = self.calculate_total_return()
        max_dd_info = self.calculate_max_drawdown()
        sharpe_ratio = self.calculate_sharpe_ratio()
        
        # Trade statistics
        num_trades = len(self.trades)
        if num_trades > 0:
            winning_trades = sum(1 for trade in self.trades if trade.get('pnl', 0) > 0)
            losing_trades = sum(1 for trade in self.trades if trade.get('pnl', 0) <= 0)
            win_rate = winning_trades / num_trades if num_trades > 0 else 0
            avg_return = np.mean([trade.get('return_pct', 0) for trade in self.trades])
        else:
            winning_trades = 0
            losing_trades = 0
            win_rate = 0
            avg_return = 0
        
        return {
            'total_return': total_return,
            'max_drawdown': max_dd_info['max_drawdown'],
            'recovery_time': max_dd_info['recovery_time'],
            'sharpe_ratio': sharpe_ratio,
            'number_of_trades': num_trades,
            'win_rate': win_rate,
            'average_return_per_trade': avg_return
        }
    
    def plot_equity_curve(self, filename='equity_curve.png'):
        """Plot the equity curve with non-interactive backend."""
        plt.figure(figsize=(12, 6))
        plt.plot(self.equity_curve)
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Equity')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        
    def plot_drawdown_curve(self, filename='drawdown_curve.png'):
        """Plot the drawdown curve with non-interactive backend."""
        # Calculate drawdown series
        peak = self.equity_curve.cummax()
        drawdown = (self.equity_curve - peak) / peak * 100
        
        plt.figure(figsize=(12, 6))
        plt.plot(drawdown)
        plt.title('Drawdown Curve')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
