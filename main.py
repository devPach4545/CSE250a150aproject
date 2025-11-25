import yfinance
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from hmmlearn import hmm

def fetch_stock_data(ticker, start_date, end_date):
    print(f"Downloading data for {ticker} from {start_date} to {end_date}...")
    data = yfinance.download(ticker, start=start_date, end=end_date)
    return data['Close']

# Using Gaussian HMM for now number of state = 3
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


# STATE: 2
def fit_hmm_model_2(prices, n_states=2):
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

# STATE: 5
def fit_hmm_model_5(prices, n_states=5):
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

# INTERPRET AND GRAPH for 5 states
def interpret_and_graph_state_5(model, hidden_states, prices, returns):
    variances = np.array([model.covars_[i][0][0] for i in range(model.n_components)])
    std_deviations = np.sqrt(variances)
    sorted_indices = np.argsort(std_deviations)
    
    map_states = {old: new for new, old in enumerate(sorted_indices)}
    new_states = []
    
    for hidden_state in hidden_states:
        new_states.append(map_states[hidden_state])
    new_states = np.array(new_states)

    colors = ['green', 'lightgreen', 'orange', 'red', 'darkred']
    plt.figure(figsize=(15, 8))
    
    dates = returns.index
    prices = prices.loc[dates]
    
    for state in range(model.n_components):
        mask = (new_states == state)
        plt.scatter(dates[mask], prices[mask], color=colors[state])
        
    plt.savefig("hmm_states_5.png")


def interpret_and_graph_state_2(model, hidden_states, prices, returns):
    variances = np.array([model.covars_[i][0][0] for i in range(model.n_components)])
    std_deviations = np.sqrt(variances)
    sorted_indices = np.argsort(std_deviations)
    
    map_states = {old: new for new, old in enumerate(sorted_indices)}
    new_states = []
    
    for hidden_state in hidden_states:
        new_states.append(map_states[hidden_state])
    new_states = np.array(new_states)

    colors = ['green', 'red']
    plt.figure(figsize=(15, 8))
    
    dates = returns.index
    prices = prices.loc[dates]
    
    for state in range(model.n_components):
        mask = (new_states == state)
        plt.scatter(dates[mask], prices[mask], color=colors[state])
        
    plt.savefig("hmm_states_2.png")



# let's make another hmm with sa

def interpret_and_graph(model, hidden_states, prices, returns):
    variances = np.array([model.covars_[i][0][0] for i in range(model.n_components)])
    std_deviations = np.sqrt(variances)
    sorted_indices = np.argsort(std_deviations)
    
    map_states = {old: new for new, old in enumerate(sorted_indices)}
    new_states = []
    
    for hidden_state in hidden_states:
        new_states.append(map_states[hidden_state])
    new_states = np.array(new_states)

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
    # We need the training data (X) to calculate scores
    log_returns = np.log(prices / prices.shift(1)).dropna()
    X = log_returns.values.reshape(-1, 1)

    print("\n" + "="*40)
    print(f"{'N_States':<10} | {'Log-Likelihood':<20}")
    print("-" * 40)

    for n in [2, 3, 5]:
        best_score = -np.inf
        
        for i in range(10):
            model, hidden_states, returns = fit_hmm_model(prices, n_states=n)
            score = model.score(X)
            if score > best_score:
                best_score = score
        
        print(f"{n:<10} | {best_score:<20.4f}")



    # Let's find the parameters for 3 state HMM
    print("="*40 + "\n")
    model, hidden_states, returns = fit_hmm_model(prices)

    print(" 3 State HMM Parameters ")
    print("="*40)

    # get model mean and var
    means = model.means_.flatten()
    variances = model.covars_.flatten()
    
    std_devs = np.sqrt(variances)
    sorted_indices = np.argsort(std_devs)

    print("\nState Parameters (sorted by volatility):")
    print(f"{'State':<10} | {'Mean':<15} | {'Variance':<15} | {'Std Dev':<15}")
    print("-" * 60)
    for i, idx in enumerate(sorted_indices):
        print(f"{i:<10} | {means[idx]:<15.6f} | {variances[idx]:<15.6f} | {std_devs[idx]:<15.6f}")
    
    print("\nTransition Matrix:")
    print("-" * 40)
    transition_matrix = model.transmat_
    for i in range(model.n_components):
        row_str = " | ".join([f"{transition_matrix[i][j]:.4f}" for j in range(model.n_components)])
        print(f"State {i}: [{row_str}]")
    



    print("="*40 + "\n")
    interpret_and_graph(model, hidden_states, prices, returns)

    #for 5 state
    model, hidden_states, returns = fit_hmm_model_5(prices)
    interpret_and_graph_state_5(model, hidden_states, prices, returns)

    #for 2 state
    model, hidden_states, returns = fit_hmm_model_2(prices)
    interpret_and_graph_state_2(model, hidden_states, prices, returns)


    # USE OF AI: Dhaivat: I HAVE USED AI TO GENERATE TABLEs, transition matrix, and formatting that.