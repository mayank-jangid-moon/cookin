import pandas as pd
import numpy as np
import argparse
from datetime import datetime
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from data_loader import DataLoader
from trade_strategy import TradingStrategy
from backtester import Backtester
from performance_analyzer import PerformanceAnalyzer

def main():
    parser = argparse.ArgumentParser(description='Run a backtest for a trading strategy.')
    parser.add_argument('--data_file', type=str, 
                        default='BTCUSDT_historical_data_1d.csv',
                        help='Path to the CSV file with historical data')
    parser.add_argument('--start_date', type=str, default='2020-01-01',
                        help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default='2023-01-01',
                        help='End date for backtest (YYYY-MM-DD)')
    parser.add_argument('--initial_capital', type=float, default=10000,
                        help='Initial capital for backtest')
    parser.add_argument('--risk_pct', type=float, default=2,
                        help='Risk percentage per trade')
    parser.add_argument('--fast_ma', type=int, default=20,
                        help='Fast moving average window')
    parser.add_argument('--slow_ma', type=int, default=50,
                        help='Slow moving average window')
    parser.add_argument('--output_dir', type=str, default='backtest_results',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    data_loader = DataLoader(args.data_file)
    data = data_loader.filter_by_date(args.start_date, args.end_date)
    
    if data.empty:
        print(f"No data found for the period {args.start_date} to {args.end_date}")
        return
    
    print(f"Data loaded: {len(data)} rows from {data.index.min()} to {data.index.max()}")
    
    # Initialize strategy
    strategy = TradingStrategy(
        fast_window=args.fast_ma,
        slow_window=args.slow_ma,
        risk_pct=args.risk_pct
    )
    
    print(f"Running backtest with {args.fast_ma}/{args.slow_ma} MA crossover strategy...")
    
    # Run backtest
    backtester = Backtester(data, strategy, args.initial_capital)
    portfolio, trades = backtester.run()
    
    # Analyze performance
    analyzer = PerformanceAnalyzer(portfolio['equity'], trades)
    summary = analyzer.generate_summary()
    
    # Save equity curve plot
    equity_plot_file = os.path.join(args.output_dir, 'equity_curve.png')
    analyzer.plot_equity_curve(equity_plot_file)
    
    # Save drawdown plot
    drawdown_plot_file = os.path.join(args.output_dir, 'drawdown_curve.png')
    analyzer.plot_drawdown_curve(drawdown_plot_file)
    
    # Save trades to CSV
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_file = os.path.join(args.output_dir, 'trades.csv')
        trades_df.to_csv(trades_file)
    
    # Print results
    print("\n==== Backtest Results ====")
    print(f"Period: {args.start_date} to {args.end_date}")
    print(f"Initial Capital: ${args.initial_capital:.2f}")
    print(f"Final Equity: ${portfolio['equity'].iloc[-1]:.2f}")
    print(f"Total Return: {summary['total_return']:.2f}%")
    print(f"Max Drawdown: {summary['max_drawdown']:.2f}%")
    print(f"Recovery Time: {summary['recovery_time']}")
    print(f"Sharpe Ratio: {summary['sharpe_ratio']:.2f}")
    print(f"Number of Trades: {summary['number_of_trades']}")
    print(f"Win Rate: {summary['win_rate']*100:.2f}%")
    print(f"Average Return per Trade: {summary['average_return_per_trade']:.2f}%")
    print(f"\nPlots and trade data saved to {args.output_dir} directory")

if __name__ == '__main__':
    main()
