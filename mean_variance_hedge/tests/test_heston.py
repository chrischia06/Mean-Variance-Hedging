# import numpy as np
# import pytest
# from mean_variance_hedge.heston import generate_CIR_paths, generate_Heston_paths


# def test_single_CIR_path():
# 	n_samples = 1
# 	S0 = 100
# 	T = 30
# 	r = 0
# 	sigma = 0.2
# 	dt = 30 / 250
# 	seed = 2021

# 	_, St_1 = generate_CIR_paths(n_samples, alpha, b, sigma, dt, v_0, T, seed)
# 	_, St_2 = generate_CIR_paths(n_samples, alpha, b, sigma, dt, v_0, T, seed)

# 	assert np.allclose(St_1, St_2)

# def test_multiple_CIR_paths():
# 	n_samples = 1
# 	S0 = 100
# 	T = 30
# 	r = 0
# 	sigma = 0.2
# 	dt = 30 / 250
# 	seed = 2021

# 	_, St_1 = generate_CIR_paths(n_samples, alpha, b, sigma, dt, v_0, T, seed)
# 	_, St_2 = generate_CIR_paths(n_samples, alpha, b, sigma, dt, v_0, T, seed)

# 	assert np.allclose(St_1, St_2)

# def test_single_Heston_path():
# 	n_samples = 1
# 	S0 = 100
# 	T = 30
# 	r = 0
# 	sigma = 0.2
# 	dt = 30 / 250
# 	seed = 2021

# 	_, St_1 = generate_CIR_paths(n_samples, alpha, b, sigma, dt, v_0, T, seed)
# 	_, St_2 = generate_CIR_paths(n_samples, alpha, b, sigma, dt, v_0, T, seed)

# 	assert np.allclose(St_1, St_2)

# def test_multiple_Heston_paths():
# 	n_samples = 1
# 	S0 = 100
# 	T = 30
# 	r = 0
# 	sigma = 0.2
# 	dt = 30 / 250
# 	seed = 2021

# 	_, St_1 = generate_CIR_paths(n_samples, alpha, b, sigma, dt, v_0, T, seed)
# 	_, St_2 = generate_CIR_paths(n_samples, alpha, b, sigma, dt, v_0, T, seed)

# 	assert np.allclose(St_1, St_2)

