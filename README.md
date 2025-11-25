# Predicting Volatility Regimes of a Stock Using Hidden Markov Models

A probabilistic approach to identifying and predicting stock market volatility regimes using Gaussian Hidden Markov Models (HMMs). This project applies the Expectation-Maximization (EM) algorithm and Viterbi decoding to analyze NVDA stock data and classify market conditions into distinct volatility states.

## Overview

Stock market volatility measures the variability of asset prices over time. High volatility indicates significant price fluctuations (higher risk), while low volatility represents stable, predictable prices (lower risk). This project uses Gaussian HMMs to:

- Model stock price log-returns as sequences with hidden volatility states
- Learn model parameters (transition probabilities, emission distributions) via EM algorithm
- Decode the most likely sequence of volatility regimes using Viterbi algorithm
- Compare models with different numbers of hidden states (2, 3, and 5 states)

## Dataset

**Source:** [yfinance (Yahoo Finance)](https://ranaroussi.github.io/yfinance/)

- **Stock:** NVIDIA (NASDAQ: NVDA)
- **Training Period:** January 1, 2022 - January 1, 2025
- **Test Period:** January 1, 2025 - November 1, 2025
- **Data Type:** Daily closing prices converted to log-returns: $x_t = \log(P_t/P_{t-1})$

## Methodology

### Gaussian Hidden Markov Model

The model consists of:

1. **Hidden States (S):** Discrete volatility regimes $s_t \in \{1,...,K\}$ where $K$ is the number of states
2. **Observations (X):** Log-returns sequence $\{x_1, x_2, ..., x_T\}$
3. **Model Parameters:**
   - **Transition Probabilities:** $a_{i,j} = P(S_{t+1}=j|S_t=i)$
   - **Emission Probabilities:** $P(x_t|S_t=j) = \mathcal{N}(x_t; \mu_j, \sigma_j^2)$
   - **Initial State Distribution:** $\pi_i = P(S_1=i)$

### Algorithms

- **EM Algorithm:** Learns HMM parameters from historical training data
- **Viterbi Algorithm:** Finds the most likely sequence of hidden states for classification
- **Multiple Random Initializations:** Runs 10 iterations per model to avoid local minima

## Installation

### Prerequisites

Python 3.7 or higher is required.

### Install Dependencies

```bash
pip install -r requirements.txt
```

Required libraries:
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `matplotlib` - Data visualization
- `yfinance` - Stock market data retrieval
- `hmmlearn` - Hidden Markov Model implementation
- `scipy` - Statistical functions

## Usage

Run the main script to train models and generate visualizations:

```bash
python main.py
```

### Output Files

The script generates the following visualizations:

- `hmm_states.png` - 3-state model results on training data
- `hmm_states_2.png` - 2-state model results
- `hmm_states_5.png` - 5-state model results
- `hmm_states_test.png` - 3-state model predictions on test data
- `latent_factor_gaussians.png` - Gaussian distributions for each volatility regime

### Console Output

The script prints:
- Log-likelihood comparison for 2, 3, and 5-state models
- Model parameters (means, variances, standard deviations) sorted by volatility
- Transition probability matrix

## Results

### Model Comparison

| Hidden States | Log-Likelihood | Interpretation |
|--------------|----------------|----------------|
| 2 | 1505.12 | Low/High volatility only |
| **3** | **1525.85** | **Low/Medium/High volatility (best)** |
| 5 | 1527.69 | Marginal improvement, higher complexity |

The **3-state model** provides the best balance between model complexity and performance, showing a significant improvement (+20 points) over the 2-state model while avoiding the diminishing returns of the 5-state model (+2 points).

### Volatility Regime Characteristics (3-State Model)

- **State 0 (Low Volatility):** $\sigma^2 \approx 0.00056$ - Stable, predictable prices
- **State 1 (Medium Volatility):** $\sigma^2 \approx 0.00157$ - Moderate fluctuations (2.8× higher variance)
- **State 2 (High Volatility):** $\sigma^2 > 0.00157$ - Significant price swings, highest risk

### Key Findings

- Successfully captured NVDA's 2022 tech crash (high volatility period)
- Identified the 2023-2024 AI-driven rally (low volatility, steady growth)
- Model generalizes well to test data (January-November 2025)

## Project Structure

```
CSE250a150aproject/
├── main.py                 # Main script for HMM training and visualization
├── requirements.txt        # Python dependencies
├── neurips_2024.tex       # Full research paper (LaTeX format)
├── README.md              # This file
└── [generated plots]      # Output visualizations (*.png)
```

## Limitations

- **Gaussian Assumption:** May not capture heavy-tailed distributions or extreme market events
- **Log-Returns Only:** Additional features (volume, sentiment) could improve predictions
- **Transition Dynamics:** HMM transitions may not perfectly mirror real market regime shifts
- **Local Optima:** EM algorithm does not guarantee global optimum despite multiple initializations

## Future Improvements

- Implement model selection criteria (AIC, BIC) for optimal state count
- Explore alternative distributions (Student's t, mixture models) for heavy-tailed returns
- Incorporate additional features beyond log-returns (trading volume, technical indicators)
- Develop trading strategies based on predicted volatility regimes
- Extend analysis to multiple tickers for portfolio risk assessment

## Contributors

- **Anthony Wang** - Research, library integration, HMM implementation, data processing
- **Anurag Chaudhari** - Methodology design, Section 3 writeup, Gaussian distribution plots
- **Rudy Osuna** - Documentation, report writing, code interpretation, project coordination
- **Lily Tagvoryan** - Results analysis, Section 5 writeup, model interpretation
- **Dhaivat Pachchigar** - Code extension (2/5-state models), testing implementation, Section 4 writeup

## References

- **Libraries:** [hmmlearn](https://hmmlearn.readthedocs.io/en/latest/), [numpy](https://numpy.org/), [pandas](https://pandas.pydata.org/), [matplotlib](https://matplotlib.org/), [yfinance](https://ranaroussi.github.io/yfinance/)
- **Dataset:** [yfinance Python API](https://ranaroussi.github.io/yfinance/)

## GitHub Repository

https://github.com/devPach4545/CSE250a150aproject

## License

This project was developed as part of CSE 250A/150A coursework.

<sub><sup>This repository used generative AI tools to assist in polishing the README and debugging implementation details.</sup></sub>
