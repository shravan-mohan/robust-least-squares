# Robust Least Squares
This function returns the optimal vectors to the robust least squares problem with factored or bounded uncertainty, based on A H Sayed's beautiful paper "A Regularized Robust Design Criterion for Uncertain Data".

# Use
N = 10
M = 10

tmp = np.random.randn(N, N)
Q = tmp @ (tmp.T)# define Q

tmp = np.random.randn(M, M)
W = tmp @ (tmp.T) # define W

A = np.random.randn(M, N)  # define A
b = np.random.randn(M, 1)  # define b

choice = 0

eta = 0.1  # define eta
etab = 0.1  # define etab

K = 5
L = 5
H = np.random.randn(M, K) / 10  # define H
Ea = np.random.randn(L, N) / 10  # define Ea
Eb = np.random.randn(L, 1) / 10 # define Eb

solver = 'ECOS'
ub_constant = 10

getOptimalVectors(N, M, Q, W, A, b, choice, H, eta, etab, K, L, Ea, Eb, ub_constant, solver)

# Package Requirements
1. Numpy
2. CVXPY
