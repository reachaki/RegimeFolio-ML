# models/mvo.py
import numpy as np


def mean_variance_weights(mu, cov, risk_aversion=1.0, l2_reg=1e-4):
    """
    Compute unconstrained mean-variance weights:
    argmax_w w^T mu - (risk_aversion/2) w^T Sigma w
    with sum(w) = 1 and small L2 regularisation for numerical stability.
    """
    n = len(mu)
    Sigma = cov + l2_reg * np.eye(n)  # regularise
    ones = np.ones(n)

    # Solve [risk_aversion * Sigma  ones] [w]   [0]
    #       [ones^T             0   ] [λ] = [1]
    A = np.block(
        [
            [risk_aversion * Sigma, ones.reshape(-1, 1)],
            [ones.reshape(1, -1), np.zeros((1, 1))],
        ]
    )
    b = np.concatenate([np.zeros(n), np.array([1.0])])

    x = np.linalg.solve(A, b)
    w = x[:n]
    return w
