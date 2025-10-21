"""
Streamlit app for stock portfolio risk analysis visualization.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import model_utils
import yfinance as yf
import io
import json

def main():
    # Set page config
    st.set_page_config(
        page_title="Portfolio Risk Analysis",
        page_icon="📈",
        layout="wide"
    )

    st.title("Stock Portfolio Risk Analysis 📈")
    st.write("Monte Carlo Simulation & Real-Time Visualization")
    
    try:
        # Load model and data
        model_obj, scaler, price_df = model_utils.load_artifacts()
        
        # Sidebar
        st.sidebar.header("Portfolio Settings")
        
        # Date range selector
        start_date = st.sidebar.date_input(
            "Start Date",
            datetime.strptime(model_obj['metadata']['start_date'], '%Y-%m-%d').date()
        )
        end_date = st.sidebar.date_input(
            "End Date",
            datetime.strptime(model_obj['metadata']['end_date'], '%Y-%m-%d').date()
        )
        
        # Stock selector
        selected_stocks = st.sidebar.multiselect(
            "Select Stocks",
            model_obj['metadata']['tickers'],
            default=model_obj['metadata']['tickers']
        )
        
        # Monte Carlo parameters
        st.sidebar.header("Simulation Parameters")
        mc_iterations = st.sidebar.slider("Monte Carlo Iterations", 1000, 20000, 10000)
        horizon_days = st.sidebar.slider("Forecast Horizon (Days)", 5, 365, 30)
        var_confidence = st.sidebar.slider("VaR Confidence Level", 0.9, 0.99, 0.95)
        
        # Portfolio weights
        st.sidebar.header("Portfolio Weights")
        use_equal_weights = st.sidebar.checkbox("Use Equal Weights", value=True)
        
        weights = []
        if not use_equal_weights:
            for stock in selected_stocks:
                weight = st.sidebar.number_input(f"{stock} Weight", 0.0, 1.0, 1.0/len(selected_stocks))
                weights.append(weight)
            
            # Normalize weights
            weights = np.array(weights) / sum(weights)
        else:
            weights = np.array([1.0/len(selected_stocks)] * len(selected_stocks))
        
        # Update live data button
        if st.sidebar.button("Update Live Data"):
            with st.spinner("Fetching latest market data..."):
                live_data = model_utils.fetch_data_yfinance(
                    selected_stocks,
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                )
                price_df = model_utils.prepare_portfolio_dataframe(live_data)
                st.success("Data updated successfully!")
        
        # Main content
        col1, col2, col3 = st.columns(3)
        
        # Key metrics
        with col1:
            st.metric("95% VaR (30-day)", 
                     f"{model_obj['performance']['var_95']:.2%}")
        with col2:
            st.metric("95% CVaR (30-day)", 
                     f"{model_obj['performance']['cvar_95']:.2%}")
        with col3:
            st.metric("Prob. of >10% Loss (30-day)", 
                     f"{model_obj['performance']['mc_stats']['prob_loss_10']:.2%}")
        
        # Historical Performance
        st.header("Historical Performance")
        fig_hist = px.line(price_df[selected_stocks], title="Portfolio Historical Prices")
        fig_hist.update_layout(showlegend=True, height=400)
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Correlation Heatmap
        st.header("Stock Correlation Matrix")
        corr_matrix = price_df[selected_stocks].pct_change().corr()
        fig_corr = px.imshow(corr_matrix, title="Correlation Heatmap",
                            color_continuous_scale='RdBu')
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Monte Carlo Simulation
        st.header("Monte Carlo Simulation")
        
        # Run new simulation with current parameters
        latest_prices = price_df[selected_stocks].iloc[-1].values
        returns_df = price_df[selected_stocks].pct_change().dropna()
        mean_returns = returns_df.mean()
        cov_matrix = returns_df.cov()
        
        portfolio_values, mc_stats = model_utils.run_monte_carlo(
            S0=latest_prices,
            mean_returns=mean_returns,
            cov_matrix=cov_matrix,
            weights=weights,
            days=horizon_days,
            iterations=mc_iterations
        )
        
        # Plot Monte Carlo histogram
        fig_mc = go.Figure()
        fig_mc.add_trace(go.Histogram(x=portfolio_values, nbinsx=50,
                                    name="Monte Carlo Distribution"))
        fig_mc.add_vline(x=-mc_stats['var_95'], line_color='red', 
                        annotation_text=f"95% VaR: {mc_stats['var_95']:.2%}")
        fig_mc.update_layout(title="Monte Carlo Simulation Results",
                            xaxis_title="30-day Return",
                            yaxis_title="Frequency")
        st.plotly_chart(fig_mc, use_container_width=True)
        
        # Future Price Predictions
        st.header("Future Price Predictions")
        prediction_days = st.slider("Prediction Horizon (Days)", 30, 365, 252)
        n_simulations = st.slider("Number of Simulation Paths", 100, 2000, 1000)
        
        if st.button("Generate Future Predictions"):
            with st.spinner("Generating future price predictions..."):
                predictions = model_utils.predict_future_prices(
                    latest_prices=latest_prices,
                    mean_returns=mean_returns.values,
                    cov_matrix=cov_matrix.values,
                    days=prediction_days,
                    simulations=n_simulations
                )
                
                # Create date index for predictions
                last_date = price_df.index[-1]
                future_dates = pd.date_range(
                    start=last_date,
                    periods=prediction_days + 1,
                    freq='B'  # Business days
                )
                
                # Plot predictions for each stock
                for i, stock in enumerate(selected_stocks):
                    st.subheader(f"Predicted Prices for {stock}")
                    
                    fig = go.Figure()
                    
                    # Plot historical data
                    hist_dates = price_df.index[-252:]  # Last year of historical data
                    hist_prices = price_df[stock].iloc[-252:]
                    fig.add_trace(go.Scatter(
                        x=hist_dates,
                        y=hist_prices,
                        name="Historical",
                        line=dict(color='blue')
                    ))
                    
                    # Plot mean prediction
                    fig.add_trace(go.Scatter(
                        x=future_dates,
                        y=predictions['mean_path'][:, i],
                        name="Mean Prediction",
                        line=dict(color='green')
                    ))
                    
                    # Plot confidence intervals
                    fig.add_trace(go.Scatter(
                        x=future_dates,
                        y=predictions['upper_95'][:, i],
                        name="95% Upper Bound",
                        line=dict(color='red', dash='dash')
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=future_dates,
                        y=predictions['lower_95'][:, i],
                        name="95% Lower Bound",
                        line=dict(color='red', dash='dash'),
                        fill='tonexty'
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        title=f"{stock} Price Prediction",
                        xaxis_title="Date",
                        yaxis_title="Price",
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display predicted values
                    final_price = predictions['mean_path'][-1, i]
                    upper_bound = predictions['upper_95'][-1, i]
                    lower_bound = predictions['lower_95'][-1, i]
                    
                    pred_return = (final_price / latest_prices[i] - 1) * 100
                    
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                    with metrics_col1:
                        st.metric(
                            "Predicted Final Price",
                            f"₹{final_price:.2f}",
                            f"{pred_return:.1f}%"
                        )
                    with metrics_col2:
                        st.metric("Upper 95% Bound", f"₹{upper_bound:.2f}")
                    with metrics_col3:
                        st.metric("Lower 95% Bound", f"₹{lower_bound:.2f}")
                
                # Save predictions to CSV
                pred_df = pd.DataFrame(
                    predictions['mean_path'],
                    columns=selected_stocks,
                    index=future_dates
                )
                
                st.download_button(
                    "Download Predictions (CSV)",
                    pred_df.to_csv(),
                    "price_predictions.csv",
                    "text/csv",
                    key='download-predictions'
                )
                
        # Download Report
        st.header("Download Report")
        
        if st.button("Generate Report"):
            # Create report data
            report_data = {
                "report_date": datetime.now().isoformat(),
                "portfolio": {
                    "stocks": selected_stocks,
                    "weights": weights.tolist()
                },
                "risk_metrics": {
                    "var_95": float(mc_stats['var_95']),
                    "cvar_95": float(mc_stats['cvar_95']),
                    "prob_loss_10": float(mc_stats['prob_loss_10']),
                    "expected_return": float(mc_stats['mean']),
                    "volatility": float(mc_stats['std'])
                }
            }
            
            # Convert to CSV
            report_df = pd.DataFrame([report_data])
            csv = report_df.to_csv(index=False)
            
            # Create download button
            st.download_button(
                "Download Report (CSV)",
                csv,
                "portfolio_risk_report.csv",
                "text/csv",
                key='download-csv'
            )
            
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please run train_model.py first to generate the required model artifacts.")

if __name__ == "__main__":
    main()

# Set page config
st.set_page_config(
    page_title="Portfolio Risk Analysis",
    page_icon="📈",
    layout="wide"
)

# Cache data loading
@st.cache_data(ttl=3600)
def load_data():
    model_obj, scaler, price_df = model_utils.load_artifacts()
    return model_obj, scaler, price_df

@st.cache_data(ttl=3600)
def fetch_live_data(tickers, start_date, end_date):
    return model_utils.fetch_data_yfinance(tickers, start_date, end_date)

def main():
    st.title("Stock Portfolio Risk Analysis 📈")
    st.write("Monte Carlo Simulation & Real-Time Visualization")
    
    try:
        # Load model and data
        model_obj, scaler, price_df = load_data()
        
        # Sidebar
        st.sidebar.header("Portfolio Settings")
        
        # Date range selector
        start_date = st.sidebar.date_input(
            "Start Date",
            datetime.strptime(model_obj['metadata']['start_date'], '%Y-%m-%d').date()
        )
        end_date = st.sidebar.date_input(
            "End Date",
            datetime.strptime(model_obj['metadata']['end_date'], '%Y-%m-%d').date()
        )
        
        # Stock selector
        selected_stocks = st.sidebar.multiselect(
            "Select Stocks",
            model_obj['metadata']['tickers'],
            default=model_obj['metadata']['tickers']
        )
        
        # Monte Carlo parameters
        st.sidebar.header("Simulation Parameters")
        mc_iterations = st.sidebar.slider("Monte Carlo Iterations", 1000, 20000, 10000)
        horizon_days = st.sidebar.slider("Forecast Horizon (Days)", 5, 365, 30)
        var_confidence = st.sidebar.slider("VaR Confidence Level", 0.9, 0.99, 0.95)
        
        # Portfolio weights
        st.sidebar.header("Portfolio Weights")
        use_equal_weights = st.sidebar.checkbox("Use Equal Weights", value=True)
        
        weights = []
        if not use_equal_weights:
            for stock in selected_stocks:
                weight = st.sidebar.number_input(f"{stock} Weight", 0.0, 1.0, 1.0/len(selected_stocks))
                weights.append(weight)
            
            # Normalize weights
            weights = np.array(weights) / sum(weights)
        else:
            weights = np.array([1.0/len(selected_stocks)] * len(selected_stocks))
        
        # Update live data button
        if st.sidebar.button("Update Live Data"):
            with st.spinner("Fetching latest market data..."):
                live_data = fetch_live_data(selected_stocks, start_date.strftime('%Y-%m-%d'), 
                                        end_date.strftime('%Y-%m-%d'))
                price_df = model_utils.prepare_portfolio_dataframe(live_data)
                st.success("Data updated successfully!")
        
        # Main content
        col1, col2, col3 = st.columns(3)
        
        # Key metrics
        with col1:
            st.metric("95% VaR (30-day)", 
                     f"{model_obj['performance']['var_95']:.2%}")
        with col2:
            st.metric("95% CVaR (30-day)", 
                     f"{model_obj['performance']['cvar_95']:.2%}")
        with col3:
            st.metric("Prob. of >10% Loss (30-day)", 
                     f"{model_obj['performance']['mc_stats']['prob_loss_10']:.2%}")
        
        # Historical Performance
        st.header("Historical Performance")
        fig_hist = px.line(price_df[selected_stocks], title="Portfolio Historical Prices")
        fig_hist.update_layout(showlegend=True, height=400)
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Correlation Heatmap
        st.header("Stock Correlation Matrix")
        corr_matrix = price_df[selected_stocks].pct_change().corr()
        fig_corr = px.imshow(corr_matrix, title="Correlation Heatmap",
                            color_continuous_scale='RdBu')
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Monte Carlo Simulation
        st.header("Monte Carlo Simulation")
        
        # Run new simulation with current parameters
        latest_prices = price_df[selected_stocks].iloc[-1].values
        returns_df = price_df[selected_stocks].pct_change().dropna()
        mean_returns = returns_df.mean()
        cov_matrix = returns_df.cov()
        
        portfolio_values, mc_stats = model_utils.run_monte_carlo(
            S0=latest_prices,
            mean_returns=mean_returns,
            cov_matrix=cov_matrix,
            weights=weights,
            days=horizon_days,
            iterations=mc_iterations
        )
        
        # Plot Monte Carlo histogram
        fig_mc = go.Figure()
        fig_mc.add_trace(go.Histogram(x=portfolio_values, nbinsx=50,
                                    name="Monte Carlo Distribution"))
        fig_mc.add_vline(x=-mc_stats['var_95'], line_color='red', 
                        annotation_text=f"95% VaR: {mc_stats['var_95']:.2%}")
        fig_mc.update_layout(title="Monte Carlo Simulation Results",
                            xaxis_title="30-day Return",
                            yaxis_title="Frequency")
        st.plotly_chart(fig_mc, use_container_width=True)
        
        # Download Report
        st.header("Download Report")
        
        if st.button("Generate Report"):
            # Create report data
            report_data = {
                "report_date": datetime.now().isoformat(),
                "portfolio": {
                    "stocks": selected_stocks,
                    "weights": weights.tolist()
                },
                "risk_metrics": {
                    "var_95": float(mc_stats['var_95']),
                    "cvar_95": float(mc_stats['cvar_95']),
                    "prob_loss_10": float(mc_stats['prob_loss_10']),
                    "expected_return": float(mc_stats['mean']),
                    "volatility": float(mc_stats['std'])
                },
                "simulation_params": {
                    "iterations": mc_iterations,
                    "horizon_days": horizon_days,
                    "confidence_level": var_confidence
                }
            }
            
            # Convert to CSV
            report_df = pd.DataFrame([report_data])
            csv = report_df.to_csv(index=False)
            
            # Create download button
        st.download_button(
            "Download Report (CSV)",
            csv,
            "portfolio_risk_report.csv",
            "text/csv",
            key='download-csv'
        )
        
        # Future Price Predictions
        st.header("Future Price Predictions")
        prediction_days = st.slider("Prediction Horizon (Days)", 30, 365, 252)
        n_simulations = st.slider("Number of Simulation Paths", 100, 2000, 1000)
        
        if st.button("Generate Future Predictions"):
            with st.spinner("Generating future price predictions..."):
                # Get the latest prices and parameters
                latest_prices = price_df[selected_stocks].iloc[-1].values
                returns_df = price_df[selected_stocks].pct_change().dropna()
                mean_returns = returns_df.mean()
                cov_matrix = returns_df.cov()
                
                # Generate predictions
                predictions = model_utils.predict_future_prices(
                    latest_prices=latest_prices,
                    mean_returns=mean_returns.values,
                    cov_matrix=cov_matrix.values,
                    days=prediction_days,
                    simulations=n_simulations
                )
                
                # Create date index for predictions
                last_date = price_df.index[-1]
                future_dates = pd.date_range(
                    start=last_date,
                    periods=prediction_days + 1,
                    freq='B'  # Business days
                )
                
                # Plot predictions for each stock
                for i, stock in enumerate(selected_stocks):
                    st.subheader(f"Predicted Prices for {stock}")
                    
                    fig = go.Figure()
                    
                    # Plot historical data
                    hist_dates = price_df.index[-252:]  # Last year of historical data
                    hist_prices = price_df[stock].iloc[-252:]
                    fig.add_trace(go.Scatter(
                        x=hist_dates,
                        y=hist_prices,
                        name="Historical",
                        line=dict(color='blue')
                    ))
                    
                    # Plot mean prediction
                    fig.add_trace(go.Scatter(
                        x=future_dates,
                        y=predictions['mean_path'][:, i],
                        name="Mean Prediction",
                        line=dict(color='green')
                    ))
                    
                    # Plot confidence intervals
                    fig.add_trace(go.Scatter(
                        x=future_dates,
                        y=predictions['upper_95'][:, i],
                        name="95% Upper Bound",
                        line=dict(color='red', dash='dash')
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=future_dates,
                        y=predictions['lower_95'][:, i],
                        name="95% Lower Bound",
                        line=dict(color='red', dash='dash'),
                        fill='tonexty'
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        title=f"{stock} Price Prediction",
                        xaxis_title="Date",
                        yaxis_title="Price",
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display predicted values
                    final_price = predictions['mean_path'][-1, i]
                    upper_bound = predictions['upper_95'][-1, i]
                    lower_bound = predictions['lower_95'][-1, i]
                    
                    pred_return = (final_price / latest_prices[i] - 1) * 100
                    
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                    with metrics_col1:
                        st.metric(
                            "Predicted Final Price",
                            f"₹{final_price:.2f}",
                            f"{pred_return:.1f}%"
                        )
                    with metrics_col2:
                        st.metric("Upper 95% Bound", f"₹{upper_bound:.2f}")
                    with metrics_col3:
                        st.metric("Lower 95% Bound", f"₹{lower_bound:.2f}")
                        
                # Save predictions to CSV
                pred_df = pd.DataFrame(
                    predictions['mean_path'],
                    columns=selected_stocks,
                    index=future_dates
                )
                
                st.download_button(
                    "Download Predictions (CSV)",
                    pred_df.to_csv(),
                    "price_predictions.csv",
                    "text/csv",
                    key='download-predictions'
                )
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please run train_model.py first to generate the required model artifacts.")

if __name__ == "__main__":
    main()
