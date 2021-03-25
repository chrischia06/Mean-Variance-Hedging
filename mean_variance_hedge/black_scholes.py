"""
Implementations of functions for 
Black-Scholes European Options Pricing

"""

import numpy as np
from numpy.random import default_rng
from scipy.stats import norm
from scipy.optimize import brentq

def generate_GBM_paths(n_samples, S0, T, r, sigma, dt, seed=2021):
    """
    Exact simulation of GBM under the risk-neutral measure Q
    
    + n_samples: Number of paths to simulate
    + S0: Initial Stock Price
    + T: Timesteps
    + r: risk free rate
    + sigma: volatility 
    + dt: Time increment, e.g. dt = 1/250 years. Ensure that sigma and dt are of the same scale
    + seed: seed for reproducibilitys
    
    Returns:

    + tis: timesteps
    + Sts: n_samples * (T + 1) numpy array , in which each row corresponds to a stock path 

    Note: prices are generated under Q such that the drift is the risk free rate r

    """
    rng = default_rng(seed)
    Zs = rng.standard_normal((n_samples, T)) # Brownian Motion increments
    Zs = np.hstack([np.zeros((n_samples, 1)), Zs])
    tis = np.arange(T + 1) #0, 1 .. T
    tis = np.tile(tis, n_samples).reshape(n_samples, T + 1) # [[0, 1.. T], [0, 1.. T]...]
    # Sample paths
    Sts = S0 * np.exp((r - 0.5 * sigma ** 2) * tis * dt + sigma * np.sqrt(dt) * np.cumsum(Zs, axis=1))
    return tis, Sts


def BlackScholes(St, K, r, sigma, tau, flag):
    """
    Inputs: 

    + St: Current Price of the stock at time t
    + K: Strike Price
    + r: risk-free rate
    + sigma: Black-Scholes implied volatility
    + tau: time-to-maturity. Ensure that sigma, tau are in the same scale.
    + flag: 1 if call, 0 if put

    Outputs:
    
    Returns the Black-Scholes price

    """
    d1 = (np.log(St/K) + (r + 0.5 * sigma ** 2) * (tau)) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    if flag == 1:
      return norm.cdf(d1) * St - np.exp(-r * (tau)) * norm.cdf(d2) * K
    else:
      return np.exp(-r * (tau)) * K * norm.cdf(-d2) - St * norm.cdf(-d1)

def delta(St, K, r, sigma, tau, flag):
    """
    Inputs: 

    + St: Current Price of the stock at time t
    + K: Strike Price
    + r: risk-fre rate
    + sigma: Black-Scholes implied volatility
    + tau: time-to-maturity. Ensure that sigma, tau are in the same scale.
    + flag: 1 if call, 0 if put

    Outputs:
    
    Returns the Black-Scholes Delta

    """
    d1 = (np.log(St/K) + (r + 0.5 * sigma ** 2) * (tau)) / (sigma * np.sqrt(tau))
    if flag == 1:
      return norm.cdf(d1)
    else:
      return -norm.cdf(-d1)

def vega(St, K, r, sigma, tau, flag):
    """
    Inputs: 

    + St: Current Price of the stock at time t
    + K: Strike Price
    + r: risk-free rate
    + sigma: Black-Scholes implied volatility
    + tau: time-to-maturity. Ensure that sigma, tau are in the same scale.
    + flag: 1 if call, 0 if put

    Outputs:
    
    Returns the Black-Scholes Vega

    """
    d1 = (np.log(St/K) + (r + 0.5 * sigma ** 2) * (tau)) / (sigma * np.sqrt(tau))
    return St * norm.pdf(d1) * np.sqrt(tau) / 100


def bsinv(price, St, K, r, tau, flag):
    """
    Inputs: 
    
    + price: price of the liability
    + St: current stock price
    + K: strike price
    + r: risk-free rate
    + tau: time to maturity
    + flag: 1 if call, 0 if put

    Outputs:
    
    + imp_vol: Black Scholes implied volatility

    """

    error = lambda s: BlackScholes(St, K, r, s, tau, flag) - price
    imp_vol = brentq(error, 1e-9, 1e+9)

    return imp_vol

