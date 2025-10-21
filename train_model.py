"""
Training script for stock portfolio risk analysis model.
Can be run in Google Colab or locally.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import model_utils

# Configuration
TICKERS = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ITC.NS']
START_DATE = '2022-01-01'
END_DATE = datetime.now().strftime('%Y-%m-%d')
RANDOM_SEED = 42

def main():
    """Main training function"""
    print("Starting portfolio model training...")
    
    # Create necessary directories
    Path('models').mkdir(exist_ok=True)
    Path('data').mkdir(exist_ok=True)
    
    # Fetch data
    print(f"Fetching data for {len(TICKERS)} stocks...")
    all_data = model_utils.fetch_data_yfinance(TICKERS, START_DATE, END_DATE)
    
    # Prepare price data
    price_df = model_utils.prepare_portfolio_dataframe(all_data)
    
    # Calculate returns and covariance
    log_returns, mean_returns, cov_matrix = model_utils.compute_returns_covariance(price_df)
    
    # Set equal weights by default
    weights = np.array([1/len(TICKERS)] * len(TICKERS))
    
    # Calculate portfolio returns
    portfolio_returns = (log_returns * weights).sum(axis=1)
    
    # Compute VaR and CVaR
    var_95, cvar_95 = model_utils.compute_var_cvar(portfolio_returns)
    
    # Run Monte Carlo simulation
    latest_prices = price_df.iloc[-1].values
    portfolio_values, mc_stats = model_utils.run_monte_carlo(
        S0=latest_prices,
        mean_returns=mean_returns,
        cov_matrix=cov_matrix,
        weights=weights,
        days=30,
        iterations=10000,
        random_state=RANDOM_SEED
    )
    
    # Create and save model artifacts
    model_obj = {
        'metadata': {
            'creation_date': datetime.now().isoformat(),
            'tickers': TICKERS,
            'start_date': START_DATE,
            'end_date': END_DATE
        },
        'model_params': {
            'mean_returns': mean_returns.to_dict(),
            'cov_matrix': cov_matrix.to_dict(),
            'weights': weights.tolist(),
            'latest_prices': latest_prices.tolist()
        },
        'performance': {
            'var_95': float(var_95),
            'cvar_95': float(cvar_95),
            'mc_stats': {k: float(v) for k, v in mc_stats.items()}
        }
    }
    
    # Fit scaler for normalization if needed
    scaler = StandardScaler()
    scaler.fit(price_df)
    
    # Save all artifacts
    model_utils.save_artifacts(model_obj, scaler, price_df)
    
    # Save results summary
    results_summary = {
        'training_date': datetime.now().isoformat(),
        'portfolio_metrics': {
            'var_95': float(var_95),
            'cvar_95': float(cvar_95),
            'prob_loss_10': float(mc_stats['prob_loss_10']),
            'expected_return_30d': float(mc_stats['mean']),
            'volatility_30d': float(mc_stats['std'])
        }
    }
    
    with open('data/training_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Print summary
    print("\nTraining completed successfully!")
    print("\nPortfolio Risk Metrics:")
    print(f"95% VaR: {var_95:.4f}")
    print(f"95% CVaR: {cvar_95:.4f}")
    print(f"Probability of >10% loss (30 days): {mc_stats['prob_loss_10']:.4f}")
    print(f"Expected 30-day return: {mc_stats['mean']:.4f}")
    print(f"30-day volatility: {mc_stats['std']:.4f}")
    
if __name__ == '__main__':
    main()
