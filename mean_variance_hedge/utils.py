"""
Utility functions for writeup
"""

import numpy as np
import pandas as pd


def diagnosis(error):
  """ 
  Inputs: 

  + error: If there are N samples, error is a N x 1 array of the terminal value of the option hedging portfolio
  
  Outputs: 

  + results: A dataframe with 1 row, consisting of the Mean-Squared-Error and CVaR at the 1%, 5%, 10% and 50% levels
  """
  one_period_MSE = np.mean(error ** 2)
  cvar_001 = -np.mean(error[error < np.quantile(error, 0.01)])
  cvar_005 = -np.mean(error[error < np.quantile(error, 0.05)])
  cvar_010 = -np.mean(error[error < np.quantile(error, 0.1)])
  cvar_050 = -np.mean(error[error < np.quantile(error, 0.5)])

  results = pd.DataFrame([[one_period_MSE, cvar_001, 
  	                       cvar_005, cvar_010, cvar_050]])
  results.columns=["MSHE", "CVar 1%", "CVaR 5%", "CVar 10%", "CVaR 50%"]

  return results

