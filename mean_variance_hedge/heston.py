"""
Implementing the exact Heston Scheme by Broadie and Kaya (2003) (WIP)

"""

import numpy as np


def generate_CIR_paths(n_samples, alpha, b, sigma, dt, v_0, T, seed = 2021):
    """
    Implements exact Cox-Ingersoll-Ross / square-root diffusion process
    based on Glasserman (2004)
    
    Exact simulation of the process V_t with dynamics given by the SDE:

    $$dV_{t} = alpha(b - V_{t}) dt + sigma sqrt(V_{t}) dW_{t}$$

    Inputs:

    + n_samples : Number of Vts to simulate
    + alpha : speed of mean reversion
    + b: mean level of variance
    + sigma: volatility of the variance
    + dt: time-increment, e.g. dt = 1 / 250 years
    + v_0: initial variance
    + T: (integer) number of time steps
    + seed: numpy seed for reproducibility

    Outputs:

    + Vts: A n_samples by (T + 1) numpy array, where in each row corresponds
    to a path V_{i, t} from 0 .. T
    """
    rng = np.random.default_rng(seed)
    
    Vts = np.zeros((n_samples, T + 1))
    Vts[:,0] = v_0
    d = 4 * b * alpha / sigma ** 2
    c = sigma ** 2  * (1 - np.exp(-alpha * dt)) / (4 * alpha)
    
    if d > 1:
        Zs = rng.random_normal(size=(n_samples, T + 1))
        chi_sqs = rng.chisquare (df = d - 1, size=(n_samples, T + 1))
    else:
        poissons = np.zeros((n_samples, T + 1))
        chi_sqs = np.zeros((n_samples, T + 1))
    
    for i in range(n_samples):
        for t in range(T):    
            l = Vts[i, t] * np.exp(-alpha * dt) / c
            if d > 1:
                Vts[i, t + 1] = c * ((Zs[i, t] + np.sqrt(l)) ** 2 + chi_sqs[i, t])
            else:
                poissons[i, t] = rng.poisson(lam = l / 2)
                chi_sqs[i, t] = rng.chisquare(df = d + 2 * poissons[i, t])
                Vts[i, t + 1] = c * chi_sqs[i, t]
    return Vts

def generate_Heston_paths(n_samples, S0, rho, r, alpha, b, sigma, dt, v_0, T, seed = 2021):
    """
    Almost Exact Heston scheme for the process with SDE
    
    $$dS_{t} = rS_{t} dt + S_{t} sqrt(V_{t}) dW_{t}$$

    $$dV_{t} = alpha(b - V_{t}) dt + sigma sqrt(V_{t}) dZ_{t}$$
    
    $$corr(dW_{t}, dZ_{t}) = rho$$

    Inputs:

    + n_samples : Number of Vtss to simulate
    + S0: Initial Stock Price
    + rho: correlation
    + r: risk-free rate
    + alpha : speed of mean reversion
    + b: mean level of variance
    + sigma: volatility of the variance
    + dt: time-increment, e.g. dt = 1 / 250 years
    + v_0: initial variance
    + T: (integer) number of time steps
    + seed: numpy seed for reproducibility

    Outputs:

    + Vts: A n_samples by T numpy array, where in each row corresponds
    to a path
    """

    rng = np.random.default_rng(seed)
    Zs = rng.standard_normal(size = (n_samples, T)) # random normal increments
    Zs = (Zs - Zs.mean(axis= 0)) / Zs.std(axis = 0)

    # pre-generate Vts
    Vts = generate_CIR_paths(n_samples = n_samples, alpha = alpha, 
                             b = b, sigma = sigma, 
                             dt = dt, v_0 = v_0, T = T, seed = seed)

    # simulate the log-price process
    log_Sts = np.zeros((n_samples, T + 1))
    log_Sts[:, 0] = np.log(S0)

    k0 = (r - rho / sigma * alpha * b) * dt
    k1 = (rho * alpha / sigma - 0.5) * dt - rho / sigma
    k2 = rho / sigma

    for i in range(T):
        log_Sts[:, i + 1] = (log_Sts[:,i] + k0 + k1 * Vts[:,i] + k2 * Vts[:,i+1] + 
                              np.sqrt((1.0 - rho ** 2) * Vts[:,i]) * np.sqrt(dt) * Zs[:, i])

    return np.exp(log_Sts), Vts



