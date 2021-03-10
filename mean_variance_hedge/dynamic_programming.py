"""
Implements methods from

Černý, Aleš, Dynamic Programming and Mean-Variance Hedging in Discrete Time 
(October 1, 2003). Applied Mathematical Finance, 2004, 11(1), 1-25, 
Available at SSRN: https://ssrn.com/abstract=561223 
or http://dx.doi.org/10.2139/ssrn.561223 
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools

from scipy.stats import norm
from scipy.special import comb

def possible_nodes(log_ret_space, T, scale_factor):
    """
    Inputs: 

    log_ret_space: array of log-returns
    T: time at maturity

    Recommendeded to space log-returns equally so they recombine,
    and scale log-returns so that they are integers
    example: [-6, -4, -2, 0, 2, 4, 6]

    Outputs: 

    Returns possible nodes as a Dictionary, 
    Indexed by time t = 0..T, with values 
    being the attainable log-returns for that time t

    """
    assert len(log_ret_space) > 0

    log_ret_space = scale_factor * log_ret_space

    attainable_nodes = {}
    attainable_nodes[0] = [0]
    for i in range(1, T + 1):
      attainable_nodes[i] = set()
      for x in log_ret_space:
        for val in attainable_nodes[i - 1]:
          if val + x not in attainable_nodes[i]:
            attainable_nodes[i].add(val + x)

    return attainable_nodes

def variance_optimal_measure(ret_space, rf, p_probs):
    """
    Inputs:
    ret_space: array of returns, i.e. exp(log_returns)
    rf: risk-free return
    p_probs: probabilitites under the physical measure

    Outputs:

    Returns quantities a, b, m, and q_probs, the latter of which is the variance optimal probabilities
    """

    assert len(ret_space) > 0
    assert len(ret_space) == len(p_probs)
    assert abs(sum(p_probs) - 1) < 0.001

    a = np.dot(p_probs, ret_space - rf) / np.dot(p_probs, (ret_space - rf) ** 2)
    b = 1 - np.dot(p_probs, (ret_space - rf)) ** 2 / np.dot(p_probs, (ret_space - rf) ** 2)
    m = (1 - a * (ret_space - rf) / b)
    q_probs = m * p_probs
    q_probs = q_probs / np.sum(q_probs) # ensure Q-probs sum to 1
    return a, b, m, q_probs

def calc_mean_value_process(attainable_nodes, S0, rf, log_ret_space, T, scale_factor, p_probs):
    """
    Inputs:

    attainable_nodes: attainable log-returns indexed by time t, from the `possible_nodes` function

    S0: Initial asset price, e.g. 100

    scale_factor: factor to divide log-return indices by, e.g. 100

    rf: risk-free return (discrete), e.g. (1.001)

    log_ret_space: array of log-returns

    T: time at maturity

    q_probs: Variance-optimal probabilities

    Output:

    Returns Hts: Mean-Value process Ht as dictionary, indexed by time t and log-returns
    """

    N_STATES = len(log_ret_space)
    log_ret_space2 = [round(x * scale_factor) for x in log_ret_space]

    Hts = {}
    Hts[T] = {}
    for x in attainable_nodes[T]:
      Sts = S0 * np.exp(x / scale_factor)
      Hts[T][x] = np.maximum(Sts -  K, 0)

    # calculate node-values iteratively in reverse
    for t in range(T - 1, -1, -1):
      Hts[t] = {}
      for x in attainable_nodes[t]:
        Htp1 = [Hts[t + 1][x + log_ret_space2[j]] for j in range(N_STATES)]
        Hts[t][x] = np.dot(q_probs, Htp1) / rf

    return Hts

def calc_dynamic_deltas(attainable_nodes, Hts, S0, rf, log_ret_space, T, scale_factor, p_probs):
    """
    Inputs:
    attainable_nodes: attainable log-returns indexed by time t, from the `possible_nodes` function

    Hts: Mean-Value process of the liability from the function `calc_mean_value_process`

    S0: Initial asset price, e.g. 100

    scale_factor: factor to divide log-return indices by, e.g. 100

    rf: risk-free return (discrete), e.g. (1.001)

    log_ret_space: array of log-returns

    T: time at maturity

    q_probs: Variance-optimal probabilities

    Hts: S0, rf, log_ret_space, T, scale_factor, q_probs

    Outputs:

    dynamic_delta: units to hedge as dictionary, indexed by time t and log returns
    """
    ret_change = ret_space - rf
    log_ret_space2 = [round(x * scale_factor) for x in log_ret_space]
    dynamic_delta = {}
    for t in range(T - 1, -1, -1):
    
        dynamic_delta[t] = {}
        
        for x in attainable_nodes[t]:
            St = S0 * np.exp(x / scale_factor)
            ht = Hts[t][x]
            ht_change = np.array([Hts[t + 1][x + log_ret_space2[j]] - rf * ht for j in range(N_STATES)])
            cov = np.dot(p_probs, ht_change * ret_change)
            qhedge_delta = cov / (St * np.dot(p_probs, (ret_change) ** 2))
            dynamic_delta[t][x] = qhedge_delta
    return dynamic_delta

            