import yfinance as yf
import numpy as np

# ================================
# 0. Set up
# ================================

# pick any tickers for a portfolio
tickers = ["AAPL", "MSFT", "ABBV", "COST", "CAT"]
data = yf.download(tickers, period = "1y") ["Close"]

missing_count = data.isna().sum()
# ---  Data check ---
print("--- data check --- ")
if data.isna().sum().sum() == 0:
    print("No data is missing")
else:
    print("Missing count")
    print(missing_count)

print (f"Obtained {len(data)} days of price data for {len(tickers)} tickers.\n")
# ================================
# 1. Monte Carlo simulation
# ================================

#Step 1: daily return calculation for each ticker
returns = data.pct_change()

#Step 2: mean return and cov matrix
mean_returns = returns.mean()
cov_matrix = returns.cov()

annual_returns = mean_returns*252
annual_cov = cov_matrix*252

#Step 3: weight assignation
num_portfolio = 20000
num_assets = len(tickers)

portfolio_weight = np.random.random((num_portfolio, num_assets))
portfolio_weight = portfolio_weight/portfolio_weight.sum(axis=1)[:,np.newaxis]

#Step 4: Calculate portfolio return and portfolio volatility
port_ret = portfolio_weight@annual_returns
port_vol = np.sqrt(np.sum((portfolio_weight@annual_cov)*portfolio_weight, axis =1))
risk_free_rate = 0.025
sharpe_ratio = (port_ret - risk_free_rate)/port_vol

#Step 5: Max Sharpe ratio calculation
max_sr = np.argmax(sharpe_ratio)

print(" --- Simulated portfolio with Max-Sharpe ratio ---")
print("portfolio summary")
print(f"Return: {port_ret[max_sr]}")
print(f"Volatility: {port_vol[max_sr]}")
print(f"Sharpe ratio: {sharpe_ratio[max_sr]}")
weight_summary = dict(zip(tickers, portfolio_weight[max_sr].round(3).tolist()))
print(f"Portfolio weight: {weight_summary}\n")

#Plot the volatility of simulated portfolio against their expected return
import matplotlib.pyplot as plt
plt.style.use("dark_background")
plt.scatter(port_vol,port_ret, c = sharpe_ratio, cmap="viridis")
plt.scatter(port_vol[max_sr], port_ret[max_sr], color ="gold", marker="*", edgecolor="white", s = 200, label = "Max_sharpe")
plt.title("Simulated Efficient Frontier")
plt.xlabel("volatility")
plt.ylabel("Mean returns")
plt.legend()
plt.show()

# =========================================================
# 2. Modern Portfolio Theory (Closed-form solution)
# =========================================================

# Here, I implement the closed-form solution to the mean-variance optimization problem
# that I learnt in my master course. Especially, I first calculated the GMV portfolio 
# and the efficient frontier by solving the minimum variance portfolio 
# under each target expected return m.

# ---2a. Global Minimum Variance (GMV portfolio) ---

# Step 1: invert the covariance matrix, build the "1" vector

V_inv = np.linalg.inv(annual_cov.values) # inverse of the cov matrix

ones = np.ones(num_assets) # a vector of 1 

# Step 2: calculate a = 1 V^-1 1

a = ones @V_inv @ ones

# Step 3: calculate the weight, expected returns and volatility of the GMV portfolio

GMV_weights = (1/a)* (V_inv @ ones)
GMV_returns = GMV_weights @ annual_returns.values
GMV_vols = np.sqrt(1/a)

print("--- GMV portfolio summary ---")

print(f"Returns: {GMV_returns}")
      
print(f"Volatility {GMV_vols}")
      
gmv_weights_summary = dict(zip(tickers, GMV_weights.round(3).tolist())) 
print(f"Portfolio Weight: {gmv_weights_summary}\n")

# Step 3: check against the alternative formula
mu = annual_returns.values
b = ones @ V_inv @ mu   # b = 1' V^-1 mu

print(GMV_returns, b / a)   # should match almost exactly

# --- 2b. minimum variance under each target m(expected return) ---

# For a given target return m, solve for the portfolio weights that
# minimize variance subject to (i) weights summing to 1, and
# (ii) expected return = m.

mu = annual_returns.values

b = ones @ V_inv @ mu   
d = mu @ V_inv @ mu      # d = mu'V^-1 mu

det = a * d - b**2      # determinant of the 2x2 system (a b; b d)

def lam(m):
    return (d-b*m)/ det

def delta (m):
    return (a*m - b)/ det

def efficient_weights (m):
    # phi(m) = lambda(m) * V^-1 * 1  +  delta(m) * V^-1 * mu
    return lam(m)* (V_inv @ ones)  + delta(m)*(V_inv@mu)

def efficient_variance (m):
    # sigma*^2(m) = lambda(m) + delta(m) * m
    return lam(m) + delta(m)*m

# setting the target return (m) acrossing the volatitlity space for plotting the frontier

targets = np.linspace(annual_returns.min(), annual_returns.max(), 200)
exact_var = np.array([efficient_variance (m) for m in targets])
exact_vol = np.sqrt(exact_var)

#plot the frontier comparing to the simulated portfolio
plt.scatter(port_vol, port_ret, c=sharpe_ratio, cmap="viridis", alpha=0.2, s = 5, label="Simulated portfolios")
plt.plot(exact_vol, targets, color="blue", linewidth=2, label="Exact frontier (closed-form)")
plt.scatter(GMV_vols, GMV_returns, color="gold", marker="D", s=120, edgecolor="white", label="GMV")
plt.scatter(port_vol[max_sr], port_ret[max_sr], color="red", marker="*", s=200, edgecolor="white", label="Max Sharpe Ratio")
plt.colorbar(label="Sharpe Ratio")
plt.xlabel("Volatility")
plt.ylabel("Expected Return")
plt.legend()
plt.show()

