#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trading Bot with Impulse Control Algorithm & Technical Indicators

This script implements a trading bot that combines an Impulse Control Algorithm for HFT Market Making
with technical indicators like RSI, ADX, ATR, Bollinger Bands, OBV, and EMA crossovers.
The bot aims to achieve high Sharpe ratio, low drawdown, and high trade frequency.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import yfinance as yf
from datetime import datetime
import argparse


class MM_Model_Parameters:
    """Market Making Model Parameters"""
    def __init__(self, lambda_m, lambda_p, kappa_m, kappa_p, delta, 
                 phi, alpha, q_min, q_max, T, cost, rebate):
        self.lambda_m = lambda_m  # Number of market order arrivals per minute on the bid
        self.lambda_p = lambda_p  # Number of market order arrivals per minute on the ask
        self.kappa_m = kappa_m    # Decay parameter for "fill rate" on the bid
        self.kappa_p = kappa_p    # Decay parameter for "fill rate" on the ask
        self.delta = delta        # Drift parameter
        self.phi = phi            # Running inventory penalty parameter
        self.alpha = alpha        # Terminal inventory penalty parameter
        self.q_min = q_min        # Minimum inventory constraint
        self.q_max = q_max        # Maximum inventory constraint
        self.T = T                # Trading horizon in minutes
        self.cost = cost          # Trading cost per market order
        self.rebate = rebate      # Rebate for providing liquidity


class AS3P_Finite_Difference_Solver:
    """Solver for Avellaneda-Stoikov style problems using finite difference methods"""
    
    @staticmethod
    def solve(parameters, N_steps=300):
        """
        Solve the HJB-QVI problem using finite differences.
        
        Args:
            parameters: Market making model parameters
            N_steps: Number of time steps
            
        Returns:
            tuple: (impulses, model) containing trading decisions and model state
        """
        # Initialize model
        model = ModelState(parameters, N_steps)
        
        # Solve the HJB-QVI problem (simplified implementation)
        # In a real implementation, this would contain the full numerical solver
        
        # Generate a simple model for demo purposes
        impulses = np.zeros((len(model.q_grid), N_steps))
        
        # Example rules (for demonstration):
        # Buy when inventory is negative and far from zero
        # Sell when inventory is positive and far from zero
        for q_idx, q in enumerate(model.q_grid):
            for t in range(N_steps):
                if q < -10:
                    impulses[q_idx, t] = 1  # Buy
                elif q > 10:
                    impulses[q_idx, t] = 2  # Sell
                else:
                    impulses[q_idx, t] = 0  # Market make
        
        # For each inventory level and time, calculate optimal spreads
        # (simplified implementation for demonstration)
        for q_idx, q in enumerate(model.q_grid):
            bid_spread = 0.01 + 0.002 * abs(q)  # Higher spread for higher inventory
            ask_spread = 0.01 + 0.002 * abs(q)
            
            if q > 0:  # If long, tighten bid spread to sell
                bid_spread *= 0.9
                ask_spread *= 1.1
            elif q < 0:  # If short, tighten ask spread to buy
                ask_spread *= 0.9
                bid_spread *= 1.1
                
            model.l_m[q_idx] = np.ones(N_steps) * bid_spread
            model.l_p[q_idx] = np.ones(N_steps) * ask_spread
            
            # Add time-dependent adjustment
            time_factor = np.linspace(1, 1.5, N_steps)
            model.l_m[q_idx] *= time_factor
            model.l_p[q_idx] *= time_factor
        
        # Value function for demonstration
        for q_idx, q in enumerate(model.q_grid):
            model.h[q_idx, :] = -parameters.alpha * q**2 - np.linspace(0, 1, N_steps) * parameters.phi * q**2
        
        return impulses, model


class ModelState:
    """Container for the model state"""
    def __init__(self, parameters, N_steps):
        self.parameters = parameters
        self.q_grid = np.arange(parameters.q_min, parameters.q_max + 1)
        self.t_grid = np.linspace(0, parameters.T, N_steps)
        
        # Optimal spreads for each inventory level
        self.l_m = {q_idx: np.zeros(N_steps) for q_idx in range(len(self.q_grid))}
        self.l_p = {q_idx: np.zeros(N_steps) for q_idx in range(len(self.q_grid))}
        
        # Value function
        self.h = np.zeros((len(self.q_grid), N_steps))


class TechnicalIndicators:
    """Technical indicators for trading signals"""
    @staticmethod
    def rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate Relative Strength Index"""
        if len(prices) <= period:
            return np.zeros_like(prices)
        
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum()/period
        down = -seed[seed < 0].sum()/period
        rs = up/down if down != 0 else np.inf
        rsi = np.zeros_like(prices)
        rsi[:period] = 100. - 100./(1. + rs)
        
        for i in range(period, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
                
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up/down if down != 0 else np.inf
            rsi[i] = 100. - 100./(1. + rs)
        return rsi
    
    @staticmethod
    def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Average Directional Index"""
        if len(close) <= period+1:
            return np.zeros_like(close), np.zeros_like(close), np.zeros_like(close)
            
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        tr = np.append(tr[0], tr)
        
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        
        plus_dm = np.zeros_like(up_move)
        plus_dm[(up_move > down_move) & (up_move > 0)] = up_move[(up_move > down_move) & (up_move > 0)]
        plus_dm = np.append(plus_dm[0], plus_dm)
        
        minus_dm = np.zeros_like(down_move)
        minus_dm[(down_move > up_move) & (down_move > 0)] = down_move[(down_move > up_move) & (down_move > 0)]
        minus_dm = np.append(minus_dm[0], minus_dm)
        
        atr = np.zeros_like(tr)
        atr[0] = tr[0]
        for i in range(1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            
        plus_di = 100 * np.zeros_like(plus_dm)
        minus_di = 100 * np.zeros_like(minus_dm)
        
        # Smooth DM
        smoothed_plus_dm = np.zeros_like(plus_dm)
        smoothed_minus_dm = np.zeros_like(minus_dm)
        smoothed_plus_dm[0] = plus_dm[0]
        smoothed_minus_dm[0] = minus_dm[0]
        
        for i in range(1, len(plus_dm)):
            smoothed_plus_dm[i] = (smoothed_plus_dm[i-1] * (period-1) + plus_dm[i]) / period
            smoothed_minus_dm[i] = (smoothed_minus_dm[i-1] * (period-1) + minus_dm[i]) / period
        
        # Calculate DI
        for i in range(len(plus_di)):
            if atr[i] != 0:
                plus_di[i] = 100 * smoothed_plus_dm[i] / atr[i]
                minus_di[i] = 100 * smoothed_minus_dm[i] / atr[i]
        
        # Calculate ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = np.zeros_like(dx)
        adx[period-1] = np.mean(dx[:period])
        
        for i in range(period, len(adx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx, plus_di, minus_di
    
    @staticmethod
    def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate Average True Range"""
        if len(close) <= period:
            return np.zeros_like(close)
            
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        tr = np.append(tr[0], tr)
        
        atr = np.zeros_like(tr)
        atr[0] = tr[0]
        for i in range(1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    @staticmethod
    def bollinger_bands(prices: np.ndarray, period: int = 20, std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands"""
        if len(prices) <= period:
            return np.zeros_like(prices), np.zeros_like(prices), np.zeros_like(prices)
            
        sma = np.zeros_like(prices)
        std = np.zeros_like(prices)
        upper_band = np.zeros_like(prices)
        lower_band = np.zeros_like(prices)
        
        for i in range(period-1, len(prices)):
            sma[i] = np.mean(prices[i-(period-1):i+1])
            std[i] = np.std(prices[i-(period-1):i+1], ddof=1)
            upper_band[i] = sma[i] + std_dev * std[i]
            lower_band[i] = sma[i] - std_dev * std[i]
        
        # Fill the first period-1 values
        sma[:period-1] = sma[period-1]
        upper_band[:period-1] = upper_band[period-1]
        lower_band[:period-1] = lower_band[period-1]
        
        return upper_band, sma, lower_band
    
    @staticmethod
    def on_balance_volume(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Calculate On Balance Volume"""
        obv = np.zeros_like(close)
        obv[0] = volume[0]
        
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv[i] = obv[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]
        return obv
    
    @staticmethod
    def ema(prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average"""
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        multiplier = 2.0 / (period + 1)
        
        for i in range(1, len(prices)):
            ema[i] = (prices[i] - ema[i-1]) * multiplier + ema[i-1]
        return ema
    
    @staticmethod
    def ema_crossover(prices: np.ndarray, fast_period: int = 12, slow_period: int = 26) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate EMA crossover signals"""
        fast_ema = TechnicalIndicators.ema(prices, fast_period)
        slow_ema = TechnicalIndicators.ema(prices, slow_period)
        
        # 1 for bullish crossover (fast above slow), -1 for bearish crossover (fast below slow), 0 otherwise
        crossover = np.zeros_like(prices)
        for i in range(1, len(prices)):
            if fast_ema[i] > slow_ema[i] and fast_ema[i-1] <= slow_ema[i-1]:
                crossover[i] = 1  # Bullish crossover
            elif fast_ema[i] < slow_ema[i] and fast_ema[i-1] >= slow_ema[i-1]:
                crossover[i] = -1  # Bearish crossover
        
        return crossover, fast_ema, slow_ema


class TradingBot:
    """Trading bot that combines Impulse Control Algorithm with technical indicators"""
    
    def __init__(self, parameters, lookback_period=100):
        """
        Initialize the trading bot
        
        Args:
            parameters: Market making model parameters
            lookback_period: Lookback period for calculating signals
        """
        self.parameters = parameters
        self.lookback_period = lookback_period
        self.reset()
        
    def reset(self):
        """Reset the trading bot state"""
        self.cash = 1000000  # Start with $1M
        self.inventory = 0
        self.trades = []
        self.pnl_history = [self.cash]
        self.inventory_history = [self.inventory]
        self.last_signal = 0
        
        # Solve the market making model
        self.impulses, self.model = AS3P_Finite_Difference_Solver.solve(self.parameters, N_steps=5*60)
    
    def generate_signals(self, data):
        """
        Generate trading signals based on technical indicators
        
        Args:
            data: DataFrame with price data (OHLCV)
            
        Returns:
            tuple: (signals, indicators) containing trading signals and indicator values
        """
        signals = np.zeros(len(data))
        
        # Calculate indicators
        rsi = TechnicalIndicators.rsi(data['close'].values)
        adx_vals, plus_di, minus_di = TechnicalIndicators.adx(
            data['high'].values, data['low'].values, data['close'].values)
        atr_vals = TechnicalIndicators.atr(
            data['high'].values, data['low'].values, data['close'].values)
        upper_bb, middle_bb, lower_bb = TechnicalIndicators.bollinger_bands(data['close'].values)
        obv = TechnicalIndicators.on_balance_volume(data['close'].values, data['volume'].values)
        ema_cross, fast_ema, slow_ema = TechnicalIndicators.ema_crossover(data['close'].values)
        
        # Compute signals
        for i in range(self.lookback_period, len(data)):
            signal = 0
            
            # RSI signals: Oversold and Overbought conditions
            if rsi[i] < 30:
                signal += 1
            elif rsi[i] > 70:
                signal -= 1
            
            # ADX signals: Strong trend detection
            if adx_vals[i] > 25:  # Strong trend
                if plus_di[i] > minus_di[i]:  # Bullish trend
                    signal += 1
                else:  # Bearish trend
                    signal -= 1
            
            # Bollinger Bands signals
            if data['close'].values[i] < lower_bb[i]:  # Price below lower band
                signal += 1
            elif data['close'].values[i] > upper_bb[i]:  # Price above upper band
                signal -= 1
            
            # OBV signals: Look for divergence between price and volume
            obv_slope = obv[i] - obv[i-5]
            price_slope = data['close'].values[i] - data['close'].values[i-5]
            if obv_slope > 0 and price_slope < 0:  # Bullish divergence
                signal += 1
            elif obv_slope < 0 and price_slope > 0:  # Bearish divergence
                signal -= 1
            
            # EMA crossover signals
            if ema_cross[i] == 1:  # Bullish crossover
                signal += 2  # Stronger signal weight
            elif ema_cross[i] == -1:  # Bearish crossover
                signal -= 2  # Stronger signal weight
                
            # Volatility adjustment based on ATR
            volatility_factor = atr_vals[i] / np.mean(atr_vals[i-20:i+1])
            if volatility_factor > 1.5:  # High volatility, reduce signal
                signal *= 0.5
            
            signals[i] = signal
            
        return signals, {
            'rsi': rsi,
            'adx': adx_vals,
            'plus_di': plus_di,
            'minus_di': minus_di,
            'atr': atr_vals,
            'upper_bb': upper_bb,
            'middle_bb': middle_bb,
            'lower_bb': lower_bb,
            'obv': obv,
            'fast_ema': fast_ema,
            'slow_ema': slow_ema
        }
    
    def get_impulse_decision(self, current_time_step, inventory):
        """
        Get impulse control decision based on the model
        
        Args:
            current_time_step: Current time step
            inventory: Current inventory level
            
        Returns:
            int: Decision (0: market make, 1: buy, 2: sell)
        """
        # Map inventory to model grid
        inventory_idx = np.argmin(np.abs(self.model.q_grid - inventory))
        
        # Get decision
        decision = self.impulses[inventory_idx, current_time_step]
        return decision
    
    def get_optimal_spread(self, current_time_step, inventory):
        """
        Get optimal bid-ask spread from the model
        
        Args:
            current_time_step: Current time step
            inventory: Current inventory level
            
        Returns:
            tuple: (bid_spread, ask_spread) optimal spreads
        """
        # Map inventory to model grid
        inventory_idx = np.argmin(np.abs(self.model.q_grid - inventory))
        
        # Get optimal spreads
        bid_spread = self.model.l_m[inventory_idx][current_time_step]
        ask_spread = self.model.l_p[inventory_idx][current_time_step]
        
        return bid_spread, ask_spread
    
    def execute_trade(self, price, size, timestamp, trade_type):
        """
        Execute a trade and update state
        
        Args:
            price: Execution price
            size: Trade size
            timestamp: Trade timestamp
            trade_type: Trade type ('buy' or 'sell')
        """
        cost = price * size
        if trade_type == 'buy':
            self.cash -= cost
            self.inventory += size
        else:  # sell
            self.cash += cost
            self.inventory -= size
            
        self.trades.append({
            'timestamp': timestamp,
            'price': price,
            'size': size,
            'type': trade_type,
            'inventory': self.inventory,
            'cash': self.cash
        })
    
    def run_strategy(self, data):
        """
        Run the trading strategy on historical data
        
        Args:
            data: DataFrame with price data (OHLCV)
            
        Returns:
            tuple: (trades_df, pnl_history, inventory_history, indicators)
        """
        self.reset()
        
        # Generate technical indicator signals
        signals, indicators = self.generate_signals(data)
        
        for i in range(self.lookback_period, len(data)):
            current_price = data['close'].iloc[i]
            timestamp = data.index[i]
            
            # Normalize time step to model time grid (0 to n)
            time_of_day = timestamp.time()
            seconds_since_open = time_of_day.hour * 3600 + time_of_day.minute * 60 + time_of_day.second
            normalized_time = min(int((seconds_since_open % 300) / 300 * (5*60)), 5*60-1)
            
            # Get impulse control decision (0: market make, 1: buy, 2: sell)
            impulse_decision = self.get_impulse_decision(normalized_time, self.inventory)
            
            # Get technical signal (-ve: sell, +ve: buy, 0: neutral)
            tech_signal = signals[i]
            
            # Combine impulse control with technical signals
            decision = 0  # Default: market make
            
            # If impulse control wants to trade
            if impulse_decision > 0:
                if impulse_decision == 1 and tech_signal >= 0:  # Both agree on buy
                    decision = 1  # Buy
                elif impulse_decision == 2 and tech_signal <= 0:  # Both agree on sell
                    decision = 2  # Sell
            else:  # Impulse control wants to market make
                # Strong technical signal can override
                if tech_signal >= 3:
                    decision = 1  # Buy
                elif tech_signal <= -3:
                    decision = 2  # Sell
            
            # Execute decision
            if decision == 1:  # Buy
                size = 1
                self.execute_trade(current_price, size, timestamp, 'buy')
            elif decision == 2:  # Sell
                size = 1
                self.execute_trade(current_price, size, timestamp, 'sell')
            else:  # Market make
                # Get optimal bid-ask spread
                bid_spread, ask_spread = self.get_optimal_spread(normalized_time, self.inventory)
                
                # Simulate limit order execution with some probability
                bid_price = current_price - bid_spread
                ask_price = current_price + ask_spread
                
                # Simulate execution with probability inversely proportional to spread
                bid_prob = np.exp(-self.parameters.kappa_m * bid_spread) / 10
                ask_prob = np.exp(-self.parameters.kappa_p * ask_spread) / 10
                
                if np.random.random() < bid_prob:
                    self.execute_trade(bid_price, 1, timestamp, 'buy')
                
                if np.random.random() < ask_prob:
                    self.execute_trade(ask_price, 1, timestamp, 'sell')
            
            # Update PnL and inventory history
            current_portfolio_value = self.cash + self.inventory * current_price
            self.pnl_history.append(current_portfolio_value)
            self.inventory_history.append(self.inventory)
        
        # Convert trades to DataFrame
        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
        
        return trades_df, self.pnl_history, self.inventory_history, indicators


class BacktestAnalyzer:
    """Analyze and visualize backtest results"""
    
    @staticmethod
    def calculate_metrics(pnl_history, trades_df=None):
        """
        Calculate trading performance metrics
        
        Args:
            pnl_history: List of portfolio values
            trades_df: DataFrame of trades
            
        Returns:
            dict: Performance metrics
        """
        if len(pnl_history) < 2:
            return {}
            
        # Convert to numpy array if it's not
        pnl_history = np.array(pnl_history)
        
        # Calculate returns
        returns = np.diff(pnl_history) / pnl_history[:-1]
        
        # Calculate metrics
        total_return = (pnl_history[-1] - pnl_history[0]) / pnl_history[0]
        daily_returns = returns  # Assuming each step is a day
        annualized_return = np.mean(daily_returns) * 252  # Assume 252 trading days
        annualized_volatility = np.std(daily_returns) * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0
        
        # Calculate drawdowns
        peak = np.maximum.accumulate(pnl_history)
        drawdown = (peak - pnl_history) / peak
        max_drawdown = np.max(drawdown)
        
        # Calculate trade metrics if available
        trade_metrics = {}
        if trades_df is not None and not trades_df.empty:
            trade_metrics = {
                'num_trades': len(trades_df),
                'trade_frequency': len(trades_df) / len(pnl_history) if len(pnl_history) > 0 else 0,
                'win_rate': np.nan  # Would need to calculate P&L per trade
            }
        
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_value': pnl_history[-1],
            'initial_value': pnl_history[0]
        }
        
        metrics.update(trade_metrics)
        return metrics
    
    @staticmethod
    def plot_performance(pnl_history, inventory_history, trades_df=None, indicators=None, data=None):
        """
        Plot trading performance and analysis charts
        
        Args:
            pnl_history: List of portfolio values
            inventory_history: List of inventory values
            trades_df: DataFrame of trades
            indicators: Dict of technical indicators
            data: DataFrame with price data
            
        Returns:
            dict: Performance metrics
        """
        metrics = BacktestAnalyzer.calculate_metrics(pnl_history, trades_df)
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 18), sharex=True)
        
        # Plot PnL
        axes[0].plot(pnl_history)
        axes[0].set_title(f'Portfolio Value\n'
                          f'Sharpe: {metrics["sharpe_ratio"]:.2f}, '
                          f'Max Drawdown: {metrics["max_drawdown"]*100:.2f}%, '
                          f'Return: {metrics["total_return"]*100:.2f}%')
        axes[0].set_ylabel('Portfolio Value ($)')
        axes[0].grid(True)
        
        # Plot Inventory
        axes[1].plot(inventory_history)
        axes[1].set_title('Inventory')
        axes[1].set_ylabel('Position Size')
        axes[1].grid(True)
        
        # Plot Drawdown
        pnl_array = np.array(pnl_history)
        peak = np.maximum.accumulate(pnl_array)
        drawdown = (peak - pnl_array) / peak
        axes[2].fill_between(range(len(drawdown)), 0, drawdown, color='red', alpha=0.3)
        axes[2].set_title('Drawdown')
        axes[2].set_xlabel('Time')
        axes[2].set_ylabel('Drawdown (%)')
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # If data and indicators are provided, plot them
        if data is not None and indicators is not None:
            # Plot price with indicators
            fig, axes = plt.subplots(4, 1, figsize=(14, 20), sharex=True)
            
            # Plot price with Bollinger Bands
            axes[0].plot(data.index, data['close'], label='Close Price')
            axes[0].plot(data.index, indicators['upper_bb'], 'r--', label='Upper BB')
            axes[0].plot(data.index, indicators['middle_bb'], 'g--', label='Middle BB')
            axes[0].plot(data.index, indicators['lower_bb'], 'r--', label='Lower BB')
            axes[0].set_title('Price with Bollinger Bands')
            axes[0].set_ylabel('Price')
            axes[0].legend()
            axes[0].grid(True)
            
            # Plot RSI
            axes[1].plot(data.index, indicators['rsi'])
            axes[1].axhline(y=70, color='r', linestyle='--')
            axes[1].axhline(y=30, color='g', linestyle='--')
            axes[1].set_title('RSI')
            axes[1].set_ylabel('RSI')
            axes[1].grid(True)
            
            # Plot ADX
            axes[2].plot(data.index, indicators['adx'], label='ADX')
            axes[2].plot(data.index, indicators['plus_di'], 'g--', label='+DI')
            axes[2].plot(data.index, indicators['minus_di'], 'r--', label='-DI')
            axes[2].axhline(y=25, color='k', linestyle='--')
            axes[2].set_title('ADX')
            axes[2].set_ylabel('ADX/DI')
            axes[2].legend()
            axes[2].grid(True)
            
            # Plot EMAs
            axes[3].plot(data.index, data['close'], label='Close Price')
            axes[3].plot(data.index, indicators['fast_ema'], 'g', label='Fast EMA')
            axes[3].plot(data.index, indicators['slow_ema'], 'r', label='Slow EMA')
            axes[3].set_title('EMA Crossovers')
            axes[3].set_ylabel('Price')
            axes[3].set_xlabel('Date')
            axes[3].legend()
            axes[3].grid(True)
            
            plt.tight_layout()
            plt.show()
        
        # Display trades on price chart if available
        if data is not None and trades_df is not None and not trades_df.empty:
            plt.figure(figsize=(14, 7))
            plt.plot(data.index, data['close'], label='Close Price')
            
            # Plot buy trades
            buy_trades = trades_df[trades_df['type'] == 'buy']
            if not buy_trades.empty:
                plt.scatter(buy_trades['timestamp'], buy_trades['price'], 
                            marker='^', color='green', s=100, label='Buy')
            
            # Plot sell trades
            sell_trades = trades_df[trades_df['type'] == 'sell']
            if not sell_trades.empty:
                plt.scatter(sell_trades['timestamp'], sell_trades['price'], 
                            marker='v', color='red', s=100, label='Sell')
            
            plt.title('Trading Signals on Price Chart')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            plt.show()
        
        # Print metrics in a table
        print("===== Performance Metrics =====")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key.replace('_', ' ').title()}: {value:.4f}")
            else:
                print(f"{key.replace('_', ' ').title()}: {value}")
                
        return metrics


def parameter_sweep(data, phi_values=None, alpha_values=None, lambda_values=None):
    """
    Perform a parameter sweep to find optimal settings
    
    Args:
        data: DataFrame with price data
        phi_values: List of phi values to test
        alpha_values: List of alpha values to test
        lambda_values: List of lambda values to test
        
    Returns:
        DataFrame: Results of parameter sweep
    """
    # Default parameter values if none provided
    phi_values = phi_values or [0.000001, 0.00001, 0.0001]
    alpha_values = alpha_values or [0.0001, 0.001, 0.01]
    lambda_values = lambda_values or [50, 100, 150]
    
    # Fixed parameters
    kappa_m = 100
    kappa_p = 100
    delta = 0
    q_min = -25
    q_max = 25
    cost = 0.005
    rebate = 0.0025
    T = 5
    
    results = []
    
    for phi in phi_values:
        for alpha in alpha_values:
            for lambda_val in lambda_values:
                print(f"Testing: phi={phi}, alpha={alpha}, lambda={lambda_val}")
                
                # Create parameters
                test_params = MM_Model_Parameters(lambda_val, lambda_val, kappa_m, kappa_p, delta,
                                         phi, alpha, q_min, q_max, T, cost, rebate)
                
                # Run backtest
                bot = TradingBot(test_params, lookback_period=20)
                trades_df, pnl_history, inventory_history, _ = bot.run_strategy(data)
                
                # Calculate metrics
                metrics = BacktestAnalyzer.calculate_metrics(pnl_history, trades_df)
                
                # Store results
                results.append({
                    'phi': phi,
                    'alpha': alpha,
                    'lambda': lambda_val,
                    'sharpe': metrics['sharpe_ratio'],
                    'max_drawdown': metrics['max_drawdown'],
                    'total_return': metrics['total_return'],
                    'num_trades': len(trades_df) if trades_df is not None and not trades_df.empty else 0
                })
                
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by Sharpe ratio
    results_df = results_df.sort_values('sharpe', ascending=False)
    
    return results_df


def main():
    """Main function to run the trading bot"""
    parser = argparse.ArgumentParser(description='High-Frequency Trading Bot')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Ticker symbol')
    parser.add_argument('--start', type=str, default='2023-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2023-06-30', help='End date (YYYY-MM-DD)')
    parser.add_argument('--interval', type=str, default='1h', help='Data interval (1m, 5m, 15m, 30m, 1h, 1d)')
    parser.add_argument('--phi', type=float, default=0.00001, help='Running inventory penalty')
    parser.add_argument('--alpha', type=float, default=0.001, help='Terminal inventory penalty')
    parser.add_argument('--lambda_val', type=float, default=100, help='Market order arrival rate')
    parser.add_argument('--optimize', action='store_true', help='Run parameter optimization')
    
    args = parser.parse_args()
    
    # Download data
    print(f"Downloading {args.ticker} data from {args.start} to {args.end} with {args.interval} interval...")
    data = yf.download(args.ticker, start=args.start, end=args.end, interval=args.interval)
    
    if len(data) == 0:
        print("No data downloaded. Check your ticker symbol and dates.")
        return
    
    print(f"Downloaded {len(data)} data points")
    print(data.head())
    
    # Set up market making parameters
    kappa_m = 100
    kappa_p = 100
    delta = 0
    q_min = -25
    q_max = 25
    cost = 0.005
    rebate = 0.0025
    T = 5
    
    if args.optimize:
        # Run parameter optimization
        print("Running parameter optimization...")
        results_df = parameter_sweep(data)
        print("\nTop 5 parameter combinations:")
        print(results_df.head(5))
        
        # Use the best parameters
        best_params = results_df.iloc[0]
        phi = best_params['phi']
        alpha = best_params['alpha']
        lambda_val = best_params['lambda']
        
        print(f"\nUsing best parameters: phi={phi}, alpha={alpha}, lambda={lambda_val}")
    else:
        # Use provided parameters
        phi = args.phi
        alpha = args.alpha
        lambda_val = args.lambda_val
    
    parameters = MM_Model_Parameters(lambda_val, lambda_val, kappa_m, kappa_p, delta,
                                    phi, alpha, q_min, q_max, T, cost, rebate)
    
    # Create and run trading bot
    print("\nRunning trading strategy...")
    bot = TradingBot(parameters, lookback_period=20)
    trades_df, pnl_history, inventory_history, indicators = bot.run_strategy(data)
    
    # Analyze results
    print("\nAnalyzing backtest results...")
    metrics = BacktestAnalyzer.plot_performance(pnl_history, inventory_history, trades_df, indicators, data)
    
    # Display trade statistics
    if not trades_df.empty:
        print(f"\nTotal trades: {len(trades_df)}")
        print(f"Buy trades: {len(trades_df[trades_df['type'] == 'buy'])}")
        print(f"Sell trades: {len(trades_df[trades_df['type'] == 'sell'])}")


if __name__ == "__main__":
    main()
