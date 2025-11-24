import yfinance
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from hmmlearn import hmm

def fetch_stock_data(ticker, start_date, end_date):
    print(f"Downloading data for {ticker} from {start_date} to {end_date}...")
    data = yfinance.download(ticker, start=start_date, end=end_date)
    return data['Close']

# Using Gaussian HMM for now
def fit_hmm_model(prices, n_states=3):
    log_returns = np.log(prices / prices.shift(1)).dropna()
    X = log_returns.values.reshape(-1, 1)
    
    best_score = -np.inf
    best_model = None

    # Try multiple random initializations to avoid local minima
    for i in range(10):
        temp_model = hmm.GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=1000, random_state=i)
        temp_model.fit(X)
    
        score = temp_model.score(X)
        if score > best_score:
            best_score = score
            best_model = temp_model

    model = best_model
    
    # Viterbi algorithm
    hidden_states = model.predict(X)
    
    return model, hidden_states, log_returns

def interpret_and_graph(model, hidden_states, prices, returns):
    variances = np.array([model.covars_[i][0][0] for i in range(model.n_components)])
    std_deviations = np.sqrt(variances)
    sorted_indices = np.argsort(std_deviations)
    
    map_states = {old: new for new, old in enumerate(sorted_indices)}
    new_states = []
    
    for hidden_state in hidden_states:
        new_states.append(map_states[hidden_state])
    new_states = np.array(new_states)
    print(hidden_states)

    colors = ['green', 'orange', 'red']
    plt.figure(figsize=(15, 8))
    
    dates = returns.index
    prices = prices.loc[dates]
    
    for state in range(model.n_components):
        mask = (new_states == state)
        plt.scatter(dates[mask], prices[mask], color=colors[state])
        
    plt.savefig("hmm_states.png")
    
if __name__ == "__main__":
    ticker = "NVDA"
    start_date = "2022-01-01"
    end_date = "2025-01-01"
    
    prices = fetch_stock_data(ticker, start_date, end_date)
    
    model, hidden_states, returns = fit_hmm_model(prices)
    interpret_and_graph(model, hidden_states, prices, returns)