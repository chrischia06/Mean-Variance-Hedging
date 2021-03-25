"""
Based on

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

    + log_ret_space: array of log-returns
    + T: time at maturity

    Outputs: 

    + attainable_nodes: Attainable nodes (log-returns) as a dictionary indexed by time t = 0, ... , T. 
    The entries of attainable_nodes[t] would be the attainable log-returns at time t

    Note: It is recommendeded to space log-returns equally so they recombine,
    and scale log-returns so that they are integers
    example: [-6, -4, -2, 0, 2, 4, 6]

    """
    assert len(log_ret_space) > 0

    log_ret_space2 = [int(round(x * scale_factor)) for x in log_ret_space]

    attainable_nodes = {}
    attainable_nodes[0] = [0]
    for i in range(1, T + 1):
      attainable_nodes[i] = set()
      for x in log_ret_space2:
        for val in attainable_nodes[i - 1]:
          if val + x not in attainable_nodes[i]:
            attainable_nodes[i].add(val + x)

    return attainable_nodes


def calc_variance_optimal_measure(ret_space, rf, p_probs):
    """
    Inputs:
    + ret_space: array of returns, i.e. exp(log_returns)
    + rf: risk-free return
    + p_probs: probabilitites under the physical measure

    Outputs:

    + a, b, m: quantities required to calculate `q_probs`
    + q_probs: the probabilities under the variance optimal probabilities
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


def calc_mean_value_process(attainable_nodes, S0, K, rf, log_ret_space, T, scale_factor, q_probs):
    """
    Inputs:

    + attainable_nodes: attainable log-returns indexed by time t, from the `possible_nodes` function

    + S0: Initial asset price, e.g. 100

    + K: Strike

    + scale_factor: factor to divide log-return indices by, e.g. 100

    + rf: risk-free return (discrete), e.g. (1.001)

    + log_ret_space: array of log-returns

    + T: time at maturity

    + q_probs: Variance-optimal probabilities

    Output:

    + Hts: Mean-Value process Ht as dictionary, indexed by time t and log-returns
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
    
    + attainable_nodes: attainable log-returns indexed by time t, from the `possible_nodes` function

    + Hts: Mean-Value process of the liability from the function `calc_mean_value_process`

    + S0: Initial asset price, e.g. 100

    + scale_factor: factor to divide log-return indices by, e.g. 100

    + rf: risk-free return (discrete), e.g. (1.001)

    + log_ret_space: array of log-returns

    + T: time at maturity

    + p_probs: physical probabilities under P

    + Hts: S0, rf, log_ret_space, T, scale_factor, q_probs

    Outputs:

    + dynamic_delta: the locally optimal hedge as a nested dictionary, indexed by time t and log-returns

    """
    log_ret_space2 = [int(round(x * scale_factor)) for x in log_ret_space]
    N_STATES = len(log_ret_space)
    ret_space = np.exp(log_ret_space)
    ret_change = ret_space - rf
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


def calc_expected_squared_replication_error(attainable_nodes, Hts, dynamic_delta, S0, rf, log_ret_space, T, scale_factor, p_probs):
    """
    Inputs:
    
    + attainable_nodes: attainable log-returns indexed by time t, from the `possible_nodes` function

    + Hts: Mean-Value process of the liability from the function `calc_mean_value_process`

    + dynamic_delta: Deltas at each node  from the function `calc_dynamic_deltas`

    + S0: Initial asset price, e.g. 100

    + scale_factor: factor to divide log-return indices by, e.g. 100

    + rf: risk-free return (discrete), e.g. (1.001)

    + log_ret_space: array of log-returns

    + T: time at maturity

    + p_probs: physical probabilities under P

    + Hts: S0, rf, log_ret_space, T, scale_factor, q_probs

    Outputs:

    + expected_squared_replication_error: the ESRE as a dictionary, indexed by time t and log returns

    """
    expected_squared_replication_error = {}
    N_STATES = len(log_ret_space)
    log_ret_space2 = [int(round(x * scale_factor)) for x in log_ret_space]
    ret_space = np.exp(log_ret_space)
    
    for t in range(T - 1, -1, -1):
        expected_squared_replication_error[t] = {}
        for x in attainable_nodes[t]:
            St = S0 * np.exp(x / scale_factor)
            ht = Hts[t][x]
            htp1 = np.array([Hts[t + 1][x + log_ret_space2[i]] for i in range(N_STATES)])
            error = rf * ht + dynamic_delta[t][x] * St * (ret_space - rf) - htp1
            expected_squared_replication_error[t][x] = np.dot(p_probs, error ** 2)

    return expected_squared_replication_error

def calc_squared_error_process(attainable_nodes, Hts, dynamic_delta, ERSEs, S0, rf, log_ret_space, T, scale_factor, p_probs, b):
    """
    Inputs:
    
    + attainable_nodes: attainable log-returns indexed by time t, from the `possible_nodes` function

    + Hts: Mean-Value process of the liability from the function `calc_mean_value_process`

    + dynamic_delta: locally optimal deltas at each node  from the function `calc_dynamic_deltas`

    + ERSEs: Expected Squared Replication Errors from `calc_expected_squared_replication_error`

    + S0: Initial asset price, e.g. 100

    + scale_factor: factor to divide log-return indices by, e.g. 100

    + rf: risk-free return (discrete), e.g. (1.001)

    + log_ret_space: array of log-returns

    + T: time at maturity

    + p_probs: physical probabilities under P

    + b: value from `calc_variance_optimal_measure`

    + Hts: S0, rf, log_ret_space, T, scale_factor, q_probs

    Outputs:

    + squared_error_process: the Squared Error Process as a nested dictionary, indexed by time t and log returns

    """
    squared_error_process = {}
    squared_error_process[T] = {}

    N_STATES = len(log_ret_space)
    log_ret_space2 = [int(round(x * scale_factor)) for x in log_ret_space]
    
    for x in attainable_nodes[T]:
        squared_error_process[T][x] = 0

    for t in range(T - 1, -1, -1):
        squared_error_process[t] = {}
        for x in attainable_nodes[t]:
            # squared error values for t + 1
            sep_tp1 = np.array([squared_error_process[t + 1][x + log_ret_space2[i]] for i in range(N_STATES)])
            # conditional expectation of squared error process for t + 1 at time t
            exp_t_sep_tp1 = np.dot(p_probs, sep_tp1)
            kt = rf ** (2 *(T - (t + 1))) * b ** (T - (t + 1))
            squared_error_process[t][x] = exp_t_sep_tp1 + kt * ERSEs[t][x]

    return squared_error_process

            