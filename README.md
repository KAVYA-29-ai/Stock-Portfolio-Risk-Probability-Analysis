# Stock Portfolio Risk Analysis with Monte Carlo Simulation

Real-time portfolio risk analysis using Monte Carlo simulation, with interactive visualization dashboard powered by Streamlit.

## Features

- Fetches real-time stock data from Yahoo Finance (Indian stocks: RELIANCE.NS, TCS.NS, HDFCBANK.NS, INFY.NS, ITC.NS)
- Calculates key risk metrics:
  - Value at Risk (VaR) at 95% confidence level
  - Conditional Value at Risk (CVaR)
  - Probability of significant losses (>10%)
- Monte Carlo simulation (10,000 iterations) for future portfolio performance
- Interactive Streamlit dashboard with Plotly visualizations
- Export functionality for risk reports
- Real-time data updates and customizable parameters

## Quick Start

### Running in Google Colab

1. Upload all files to your Google Drive
2. Open the notebook in Colab and mount your Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

3. Install requirements:
```bash
!pip install -r requirements.txt
```

4. Train the model:
```bash
!python train_model.py
```

5. Launch Streamlit app (after downloading artifacts):
```bash
!streamlit run app.py
```

### Local Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/stock-portfolio-risk-analysis.git
cd stock-portfolio-risk-analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train the model:
```bash
python train_model.py
```

4. Run the Streamlit app:
```bash
streamlit run app.py
```

### Deployment on Hugging Face Spaces

1. Create a new Space on Hugging Face (https://huggingface.co/spaces)
2. Choose "Streamlit" as the SDK
3. Link your GitHub repository
4. Add the following to your Space's settings:
   - Python dependencies: Copy contents of requirements.txt
   - Repository URL: Your GitHub repo URL
   - Additional environment variables (if needed)

## Project Structure

```
stock-portfolio-risk-analysis/
├── app.py                    # Streamlit dashboard application
├── train_model.py           # Model training script
├── model_utils.py           # Helper functions and utilities
├── requirements.txt         # Project dependencies
├── README.md               # Project documentation
├── .gitignore             # Git ignore rules
├── models/                # Saved model artifacts
│   ├── portfolio_model.pkl   
│   └── scaler.pkl            
└── data/                  # Data directory
    └── portfolio_data.csv    
```

## Example Usage

After running the model training:

```python
# Sample risk metrics output:
95% VaR: -0.0842 (8.42% maximum loss with 95% confidence)
95% CVaR: -0.1123 (11.23% expected loss in worst 5% scenarios)
Probability of >10% loss (30 days): 0.0891 (8.91% chance)
```

## Troubleshooting

1. Yahoo Finance data issues:
   - Ensure internet connectivity
   - Verify ticker symbols are correct (use .NS suffix for NSE stocks)
   - Try reducing data fetch frequency

2. Model artifacts not found:
   - Run train_model.py first
   - Check models/ directory exists
   - Verify write permissions

3. Streamlit app issues:
   - Confirm all dependencies are installed
   - Check port availability
   - Verify data files are in correct locations

## Contributing

Pull requests are welcome! For major changes, please open an issue first.

## License

MIT

## Acknowledgments

- yfinance for market data access
- Streamlit for the interactive dashboard framework
- Plotly for interactive visualizations
