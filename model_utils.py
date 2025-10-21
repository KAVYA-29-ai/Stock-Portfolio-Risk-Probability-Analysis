"""
Helper functions for stock portfolio risk analysis using Monte Carlo simulation.
Contains functions for data fetching, processing, and model management.
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
import joblib
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Union

def fetch_data_yfinance(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """
    Fetch stock data from Yahoo Finance for multiple tickers.
    
    Args:
        tickers (List[str]): List of stock ticker symbols
        start (str): Start date in YYYY-MM-DD format
        end (str): End date in YYYY-MM-DD format
        
    Returns:
        pd.DataFrame: DataFrame with stock data, with ticker symbols as columns
    """
    # Download data for all tickers at once
    try:
        data = yf.download(tickers, start=start, end=end, group_by='column', auto_adjust=True)
        
        if isinstance(data.columns, pd.MultiIndex):
            # If we have a multi-index, just keep the Close prices
            close_prices = data['Close']
        else:
            # If single ticker, construct DataFrame with Close price
            close_prices = pd.DataFrame(data['Close'])
            close_prices.columns = [tickers[0]]
        
        # Check if we got data for all tickers
        missing_tickers = set(tickers) - set(close_prices.columns)
        if missing_tickers:
            print(f"Warning: No data found for tickers: {missing_tickers}")
        
        if close_prices.empty or len(close_prices.columns) == 0:
            raise ValueError("No data retrieved for any ticker")
            
        return close_prices
        
    except Exception as e:
        raise ValueError(f"Error fetching data: {str(e)}")

def prepare_portfolio_dataframe(all_data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the data by handling missing values.
    
    Args:
        all_data (pd.DataFrame): DataFrame with closing prices
        
    Returns:
        pd.DataFrame: Clean DataFrame with closing prices
    """
    # Handle missing values
    clean_data = all_data.fillna(method='ffill').fillna(method='bfill')
    
    return clean_data

def compute_returns_covariance(price_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Compute daily log returns, mean returns, and covariance matrix.
    
    Args:
        price_df (pd.DataFrame): DataFrame of closing prices
        
    Returns:
        Tuple containing:
        - pd.DataFrame: Daily log returns
        - pd.Series: Mean returns
        - pd.DataFrame: Covariance matrix
    """
    # Calculate daily log returns
    log_returns = np.log(price_df / price_df.shift(1))
    log_returns = log_returns.dropna()
    
    # Calculate mean returns and covariance matrix
    mean_returns = log_returns.mean()
    cov_matrix = log_returns.cov()
    
    return log_returns, mean_returns, cov_matrix

def compute_var_cvar(portfolio_returns: pd.Series, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Compute Value at Risk (VaR) and Conditional Value at Risk (CVaR) using historical method.
    
    Args:
        portfolio_returns (pd.Series): Historical portfolio returns
        alpha (float): Confidence level (default: 0.05 for 95% confidence)
        
    Returns:
        Tuple containing:
        - float: VaR at specified confidence level
        - float: CVaR at specified confidence level
    """
    # Sort returns from worst to best
    sorted_returns = sorted(portfolio_returns)
    
    # Find the index for VaR
    index = int(np.ceil(alpha * len(sorted_returns))) - 1
    
    # Calculate VaR
    var = -sorted_returns[index]
    
    # Calculate CVaR (average of losses beyond VaR)
    cvar = -np.mean(sorted_returns[:index+1])
    
    return var, cvar

def run_monte_carlo(S0: np.array, mean_returns: np.array, cov_matrix: np.array,
                   weights: np.array, days: int = 30, iterations: int = 10000,
                   random_state: int = 42) -> Tuple[np.array, Dict]:
    """
    Run Monte Carlo simulation for portfolio returns.
    
    Args:
        S0 (np.array): Initial stock prices
        mean_returns (np.array): Mean returns for each stock
        cov_matrix (np.array): Covariance matrix of returns
        weights (np.array): Portfolio weights
        days (int): Number of days to simulate
        iterations (int): Number of simulation iterations
        random_state (int): Random seed for reproducibility
        
    Returns:
        Tuple containing:
        - np.array: Array of simulated final portfolio returns
        - Dict: Summary statistics of the simulation
    """
    np.random.seed(random_state)
    
    # Initialize array for final portfolio values
    portfolio_values = np.zeros(iterations)
    
    # Cholesky decomposition for correlated random variables
    L = np.linalg.cholesky(cov_matrix)
    
    # Run simulations
    for i in range(iterations):
        # Generate correlated random returns
        Z = np.random.standard_normal(size=(days, len(S0)))
        corr_returns = np.dot(Z, L.T)
        
        # Add drift term
        daily_returns = mean_returns.values + corr_returns
        
        # Calculate path
        path = np.zeros((days + 1, len(S0)))
        path[0] = S0
        
        for t in range(1, days + 1):
            path[t] = path[t-1] * np.exp(daily_returns[t-1])
        
        # Calculate final portfolio value
        final_portfolio_value = np.sum(path[-1] * weights)
        initial_portfolio_value = np.sum(S0 * weights)
        
        # Store the return
        portfolio_values[i] = (final_portfolio_value / initial_portfolio_value) - 1
    
    # Calculate summary statistics
    summary_stats = {
        'mean': np.mean(portfolio_values),
        'std': np.std(portfolio_values),
        'var_95': -np.percentile(portfolio_values, 5),
        'cvar_95': -np.mean(portfolio_values[portfolio_values <= -np.percentile(portfolio_values, 5)]),
        'prob_loss_10': np.mean(portfolio_values < -0.10)
    }
    
    return portfolio_values, summary_stats

def predict_future_prices(latest_prices: np.array, mean_returns: np.array, cov_matrix: np.array,
                         days: int = 252, simulations: int = 1000, random_state: int = 42,
                         annualization_factor: float = 252) -> Dict:
    """
    Predict future stock prices using Monte Carlo simulation with GBM.
    
    Args:
        latest_prices (np.array): Latest available stock prices
        mean_returns (np.array): Mean returns for each stock
        cov_matrix (np.array): Covariance matrix of returns
        days (int): Number of days to predict into future
        simulations (int): Number of simulation paths
        random_state (int): Random seed for reproducibility
        
    Returns:
        Dict: Dictionary containing predicted prices and confidence intervals
    """
    np.random.seed(random_state)
    
    # Time increment (1 for daily)
    dt = 1
    
    # Number of stocks
    n_stocks = len(latest_prices)
    
    # Initialize price paths array
    prices = np.zeros((simulations, days + 1, n_stocks))
    prices[:, 0] = latest_prices
    
    # Annualize parameters for long-term predictions
    ann_mean_returns = mean_returns * annualization_factor
    ann_cov_matrix = cov_matrix * annualization_factor
    
    # Cholesky decomposition for correlated random variables
    L = np.linalg.cholesky(cov_matrix)
    
    # Calculate drift and volatility terms
    drift = ann_mean_returns - 0.5 * np.diag(ann_cov_matrix)
    
    for t in range(1, days + 1):
        # Generate correlated random returns
        Z = np.random.standard_normal(size=(simulations, n_stocks))
        corr_returns = np.dot(Z, L.T)
        
        # Calculate price paths using Geometric Brownian Motion with time scaling
        time_scaling = t / annualization_factor
        prices[:, t] = prices[:, t-1] * np.exp(
            drift * time_scaling + 
            np.sqrt(time_scaling) * corr_returns
        )
    
    # Calculate statistics for each stock
    predictions = {
        'mean_path': np.mean(prices, axis=0),
        'upper_95': np.percentile(prices, 95, axis=0),
        'lower_95': np.percentile(prices, 5, axis=0),
        'all_paths': prices  # Store some paths for visualization
    }
    
    return predictions

def save_artifacts(model_obj: Dict, scaler: object, price_df: pd.DataFrame) -> None:
    """
    Save model artifacts to disk.
    
    Args:
        model_obj (Dict): Dictionary containing model metadata and parameters
        scaler (object): Fitted scaler object
        price_df (pd.DataFrame): Price data DataFrame
    """
    # Create directories if they don't exist
    Path('models').mkdir(exist_ok=True)
    Path('data').mkdir(exist_ok=True)
    
    # Save model object
    joblib.dump(model_obj, 'models/portfolio_model.pkl')
    
    # Save scaler if provided
    if scaler is not None:
        joblib.dump(scaler, 'models/scaler.pkl')
    
    # Save price data
    price_df.to_csv('data/portfolio_data.csv')

def load_artifacts() -> Tuple[Dict, object, pd.DataFrame]:
    """
    Load saved model artifacts from disk.
    
    Returns:
        Tuple containing:
        - Dict: Model object with metadata and parameters
        - object: Fitted scaler object
        - pd.DataFrame: Price data DataFrame
    """
    try:
        model_obj = joblib.load('models/portfolio_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        price_df = pd.read_csv('data/portfolio_data.csv', index_col=0, parse_dates=True)
        return model_obj, scaler, price_df
    except Exception as e:
        raise RuntimeError(f"Error loading artifacts: {str(e)}")
