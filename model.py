import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt

# ==========================
# 1. CONFIG
# ==========================
#TICKERS = ["AAPL.US", "MSFT.US", "GOOG.US", "SPY.US", "TLT.US"]  # mix rising/falling
#TICKERS = ["IBM.US", "INTC.US", "GE.US", "XLE.US", "TLT.US"]
'''TICKERS = [
    "F.US",   # long-duration bonds
    "GE.US",   # rate sensitivity
    "INTC.US",  # secular decline
    "IBM.US",   # value trap
    "XRX.US"    # legacy business
]'''
TICKERS = [
    "SQQQ.US",   # 3x inverse Nasdaq
    "SPXS.US",   # 3x inverse S&P 500
    "UVXY.US"    # VIX decay product
]
START = "2020-01-01"
END = "2024-01-01"

N_PARTICLES = 1000
Q_PF = 0.01
EVENT_EPS = 0.005
N_LAGS = 5  # number of lagged returns for trend

np.random.seed(42)

# ==========================
# 2. DATA FETCH
# ==========================
def get_stock_data(ticker, start, end):
    data = pdr.DataReader(ticker, "stooq", start, end)
    data = data.sort_index()
    prices = data["Close"] if "Close" in data.columns else data["close"]
    return prices.dropna()

# ==========================
# 3. EVENT-TIME RETURNS
# ==========================
def event_time_returns(prices, eps):
    event_prices = [prices.iloc[0]]
    last_price = prices.iloc[0]
    for i in range(1, len(prices)):
        if abs(np.log(prices.iloc[i]) - np.log(last_price)) >= eps:
            event_prices.append(prices.iloc[i])
            last_price = prices.iloc[i]
    event_prices = np.array(event_prices)
    returns_event = np.diff(np.log(event_prices), prepend=np.log(event_prices[0]))
    return returns_event

# ==========================
# 4. PARTICLE FILTER
# ==========================
def particle_filter(returns_event):
    particles = np.random.normal(-3.0, 0.5, size=N_PARTICLES)
    weights = np.ones(N_PARTICLES) / N_PARTICLES
    x_pf = np.zeros(len(returns_event))
    x_pf_var = np.zeros(len(returns_event))
    for t in range(len(returns_event)):
        particles += np.random.normal(0, np.sqrt(Q_PF), size=N_PARTICLES)
        sigma = np.exp(particles)
        likelihood = (1/np.sqrt(2*np.pi*sigma**2)) * np.exp(-0.5*(returns_event[t]/sigma)**2)
        weights *= likelihood
        weights += 1e-300
        weights /= np.sum(weights)
        mean = np.sum(weights * particles)
        var = np.sum(weights * (particles - mean)**2)
        x_pf[t] = mean
        x_pf_var[t] = var
        idx = np.random.choice(N_PARTICLES, size=N_PARTICLES, p=weights)
        particles = particles[idx]
        weights.fill(1.0/N_PARTICLES)
    return x_pf, x_pf_var

# ==========================
# 5. FEATURE ENGINEERING
# ==========================
def build_features(x_pf, x_pf_var, returns_event, n_lags=N_LAGS):
    T = len(returns_event) - 1  # number of feature rows

    # Particle filter features
    v = np.diff(x_pf, prepend=x_pf[0])[:T]
    a = np.diff(v, prepend=v[0])[:T]
    energy = 0.5 * v**2

    # Lagged returns
    lagged_returns = np.zeros((T, n_lags))
    for lag in range(1, n_lags+1):
        lagged_returns[lag:, lag-1] = returns_event[1:T-lag+1]

    # Rolling momentum
    momentum = np.array([returns_event[max(0, i-n_lags+1):i+1].mean() for i in range(1, T+1)])

    # Combine all features
    X = np.column_stack([
        x_pf[:-1], v, a, energy, x_pf_var[:-1],
        lagged_returns, momentum
    ])

    y = returns_event[1:]
    rets = returns_event[1:]
    latent_var = x_pf_var[1:]

    return X, y, rets, latent_var

# ==========================
# 6. BUILD DATASET
# ==========================
X_train_list, y_train_list = [], []
test_data = []

for ticker in TICKERS:
    prices = get_stock_data(ticker, START, END)
    returns_event = event_time_returns(prices, EVENT_EPS)
    x_pf, x_pf_var = particle_filter(returns_event)
    X, y, rets, latent_var = build_features(x_pf, x_pf_var, returns_event)
    split = int(0.7*len(X))
    X_train_list.append(X[:split])
    y_train_list.append(y[:split])
    test_data.append({
        "X_test": X[split:],
        "rets_test": rets[split:],
        "latent_var": latent_var[split:]
    })

X_train = np.vstack(X_train_list)
y_train = np.concatenate(y_train_list)

# ==========================
# 7. TRAIN MODEL
# ==========================
model = GradientBoostingRegressor(n_estimators=300, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# ==========================
# 8. PORTFOLIO BACKTEST
# ==========================
strategy_returns, buy_hold_returns = [], []

min_len = min(len(d["rets_test"]) for d in test_data)

for d in test_data:
    X_test = d["X_test"][:min_len]
    rets = d["rets_test"][:min_len]
    latent_var = d["latent_var"][:min_len]

    mu_hat = model.predict(X_test)  # expected return
    exposure = mu_hat / np.sqrt(latent_var)  # risk-adjusted exposure
    exposure = np.clip(exposure, -1, 1)

    strategy_returns.append(exposure * rets)
    buy_hold_returns.append(rets)

strategy_portfolio = np.mean(strategy_returns, axis=0)
bh_portfolio = np.mean(buy_hold_returns, axis=0)

cum_strategy = np.cumsum(strategy_portfolio)
cum_bh = np.cumsum(bh_portfolio)

# ==========================
# 9. PLOT
# ==========================
plt.figure(figsize=(12,5))
plt.plot(cum_bh, label="Buy & Hold (Equal-Weight)", alpha=0.6)
plt.plot(cum_strategy, label="PF + Dynamics Portfolio (GBR Expected Return)", linewidth=2)
plt.title("Capital-Decline Portfolio: Strategy vs Buy & Hold")
plt.xlabel("Event Index (Test)")
plt.ylabel("Cumulative Log Return")
plt.legend()
plt.tight_layout()
plt.show()
