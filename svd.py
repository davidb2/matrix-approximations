import pandas as pd
import numpy as np
import scipy.linalg as la

from scipy.stats.distributions import norm


def discretize(xs):
  '''Converts a float tuple into an int tuple by taking the ceiling.'''
  return tuple(np.int32(np.ceil(x)) for x in xs)

def project_rows_to_subspace(A, W_basis):
  P = W_basis @ W_basis.T
  return (A @ W_basis) @ W_basis.T

def project_rows_to_rowspan(A, B):
  Q, _ = la.qr(B.T, mode='economic')  # Orthonormal basis for row space of B
  return project_rows_to_subspace(A, Q)

def best_rank_k_approximation(A, k):
  U, s, VT = la.svd(A, full_matrices=False)
  s_k, U_k, VT_k = s[:k], U[:, :k], VT[:k, :]
  return U_k, s_k, VT_k

def proj_B_k(A, B, k):
  """Computes Π_{B,k}(A), the best rank-k approximation of A with rows in rowspan(B)."""
  A_projected = project_rows_to_rowspan(A, B)
  return best_rank_k_approximation(A_projected, k)


def svd(A, k, epsilon=0.25, delta=0.25, loss=False):
  assert 0 < epsilon < 1, f'epsilon must be in (0, 1). Found epsilon={epsilon}'
  assert 0 < delta < 1, f'delta must be in (0, 1). Found delta={delta}'
  m, n = A.shape
  assert m <= n, f'need m <= n. Found {m=}, {n=}'
  assert 0 < k <= min(n, m), f'k must be in (0, min(A.shape)). Found k={k}'

  # Set r = \Theta(k / eps)
  r = k / epsilon

  # Set a JLT from R^n to R^r.
  N = norm(0, 1)
  beta = np.log(1/delta)

  # TODO: this can be space-optimized using a streaming algorithm.
  # Compute tug-of-war matrices.
  Ss = [
    (1/np.sqrt(r)) * N.rvs(size=discretize((r, m)))
    for i in range(discretize((beta,))[0])
  ]

  # Compute SA for each copy of S.
  SAs = [S@A for S in Ss]
  PAs = [proj_B_k(A, SA, k) for SA in SAs]

  # Find the best x.
  Z = np.array([np.sum(s_k**2) for _, s_k, _ in PAs])
  i = np.argmin(Z)

  return (PAs[i], Z[i]) if loss else PAs[i
