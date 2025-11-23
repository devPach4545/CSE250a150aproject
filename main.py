import yfinance
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from hmmlearn import hmm

def fetch_stock_data(ticker, start_date, end_date):
    data = yfinance.download(ticker, start=start_date, end=end_date)
    return data['Close']

# Using Gaussian HMM for now
def fit_hmm_model(prices, n_states=3):
    log_returns = np.log(prices / prices.shift(1)).dropna()
    X = log_returns.values.reshape(-1, 1)
    
    model = hmm.GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=10000)
    
    # Run EM algorithm to fit data
    model.fit(X)
    
    # Viterbi algorithm
    hidden_states = model.predict(X)
    
    return model, hidden_states

if __name__ == "__main__":
    ticker = "NVDA"
    start_date = "2022-01-01"
    end_date = "2025-01-01"
    
    prices = fetch_stock_data(ticker, start_date, end_date)
    
    model, hidden_states = fit_hmm_model(prices)