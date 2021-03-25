import numpy as np
import pytest
from mean_variance_hedge.black_scholes import generate_GBM_paths, BlackScholes, delta, vega, bsinv


def test_single_GBM_path():
	n_samples = 1
	S0 = 100
	T = 30
	r = 0
	sigma = 0.2
	dt = 30 / 250
	seed = 2021

	_, St_1 = generate_GBM_paths(n_samples, S0, T, r, sigma, dt, seed=seed)
	_, St_2 = generate_GBM_paths(n_samples, S0, T, r, sigma, dt, seed=seed)

	assert np.allclose(St_1, St_2)

def test_multiple_GBM_paths():
	n_samples = 100
	S0 = 100
	T = 30
	r = 0
	sigma = 0.2
	dt = 30 / 250
	seed = 2021

	_, St_1 = generate_GBM_paths(n_samples, S0, T, r, sigma, dt, seed=seed)
	_, St_2 = generate_GBM_paths(n_samples, S0, T, r, sigma, dt, seed=seed)

	assert np.allclose(St_1, St_2)

def test_bsinv():
	St = 100
	K = 100
	r = 0
	sigma = 0.2
	tau = 30 / 250
	flag = 1

	BS_price = BlackScholes(St, K, r, sigma, tau, flag)
	imp_vol = bsinv(BS_price, St, K, r, tau, flag)
	
	assert np.isclose(imp_vol, sigma)

