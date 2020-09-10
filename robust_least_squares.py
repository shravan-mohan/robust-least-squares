import numpy as np
import cvxpy as cvx

def getOptimalVectors(N,M,Q,W,A,b,choice,H,eta,etab,K,L,Ea,Eb,ub_constant,solver):
    """
    This function returns the optimal vectors to the robust optimization problem
    with factored or bounded uncertainty, based on A H Sayed's beautiful paper
    "A REGULARIZED ROBUST DESIGN CRITERION FOR UNCERTAIN DATA"
    :param N: Dimension of Q (see paper)
    :param M: Dimension of W (see paper)
    :param Q: Matrix Q (see paper)
    :param W: Matrix W (see paper)
    :param A: Matrix A of dimension (M,N) (see paper)
    :param b: Vector b of dimension (M) (see paper)
    :param choice: Choose 0 for bounded uncertainty and 1 for factored uncertainty
    :param H: Matrix H of dimension (M, K) (see paper)
    :param eta: Regularization parameter (see paper)
    :param etab: Regularization parameter (see paper)
    :param K: Dimension of H (see paper)
    :param L: Dimension of H (see paper)
    :param Ea: Regularization parameter (see paper)
    :param Eb: Regularization parameter (see paper)
    :param ub_constant: Multiple greater than 2 to be used as an upper bound for lambda
    :param solver: One of CVXPY solvers. Default is 'ECOS'
    :return: Returns the optimal vectors x and y
    """
    if (choice == 0):
        H = np.eye(M)

    l = np.linalg.norm((H.T)@W@H,2)
    Lower_Bound = l
    u = ub_constant * np.linalg.norm((H.T)@W@H,2) # 2 is a guess! You might have to think of other ways.
    Upper_Bound = u
    gr = (np.sqrt(5) - 1) / 2

    if (choice == 0):
        f = lambda x: caculateG_for_bounded_uncertainty(x, N, M, Q, A, b, W, H, eta, etab)
    else:
        f = lambda x: caculateG_for_factored_uncertainty(x, N, M, K, L, Q, A, b, W, H, Ea, Eb)

    TMP = max(f(Upper_Bound), f(Lower_Bound))
    err = TMP
    while (err >= (1e-6) * TMP):
        d = (u - l) * gr
        x1 = l + d
        x2 = u - d
        fx1 = f(x1)
        fx2 = f(x2)
        if (fx1 <= fx2):
            l = x2
        else:
            u = x1
        err = np.abs(fx1 - fx2)

    lambda_star = (x1 + x2) / 2

    x = cvx.Variable(N)
    if (choice == 0):
        obj = cvx.quad_form(x, Q) + \
              cvx.quad_form(A@x-b.flatten(), W + W@H@np.linalg.pinv(lambda_star*np.eye(M)-(H.T)@W@H)@(H.T)@W) + \
              lambda_star*(eta**2)*cvx.quad_form(x, np.eye(M)) + \
              2*lambda_star*eta*etab*cvx.norm2(x) + \
              lambda_star*(etab**2)
    else:
        obj = cvx.quad_form(x, Q) + \
              cvx.quad_form(A@x-b, W + W@H@np.linalg.pinv(lambda_star*np.eye(M)-(H.T)@W@H)@(H.T)@W) + \
              cvx.quad_form(Ea@x-Eb,np.eye(L))

    prob = cvx.Problem(cvx.Minimize(obj))
    prob.solve(solver=solver)
    x_star = x.value

    print('The optimal vector x is ' + str(x_star))

    return x_star

def caculateG_for_factored_uncertainty(lam , N, M, K, L, Q, A, b, W, H, Ea, Eb):

    x = cvx.Variable(N)
    obj = cvx.quad_form(x, Q) + cvx.quad_form(A @ x - b, W + (W @ H @ (np.linalg.pinv(lam * np.eye(K) - (H.T)@W@H))@(H.T) @ W)) + \
          lam*cvx.quad_form(Ea @ x - Eb, np.eye(L))
    prob = cvx.Problem(cvx.Minimize(obj))
    prob.solve(solver=solver)

    return prob.value


def caculateG_for_bounded_uncertainty(lam, N, M, Q, A, b, W, H, eta, etab):

    x = cvx.Variable(N)
    obj = cvx.quad_form(x, Q) + cvx.quad_form(A @ x - b.flatten(), W + (W @ H @ (np.linalg.pinv(lam * np.eye(M) - (H.T)@W@H))@(H.T) @ W)) + \
          lam*(eta**2)*cvx.quad_form(x, np.eye(M)) + 2*lam*eta*etab*cvx.norm2(x) + lam*(etab**2)
    prob = cvx.Problem(cvx.Minimize(obj))
    prob.solve(solver=solver)

    return prob.value


if(__name__=='__main__'):
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