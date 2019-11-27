#!/usr/bin/env python3.6
import argparse
import numpy as np

from scipy.stats.distributions import norm

def discretize(xs):
  '''Converts a float tuple into an int tuple by taking the ceiling.'''
  return tuple(np.int32(np.ceil(x)) for x in xs)


def mult(A, B, epsilon=0.25, delta=0.05, decompose=False):
  '''
  Matrix multiplication using Algorithm 1 (Page 5)
  without the use of the Fast Johnson-Lindenstrauss Transform (FJLT).

  Let C bet the output of this function. Then:
     Pr(||AB-C||_F <= 2*epsilon*||A||_F * ||B||_F) >= 1 - delta.

  Time complexity of:
    O(b(1/e^2((n+g)(m+p)) + g(n(m+p)+lg(g)))), where:
    * e = epsilon
    * b = beta = O(lg(1/delta))
    * g = gamma = O(b + lg(b)) = O(lg(1/delta) + lg(lg(1/delta))).
  with
    * epsilon > 0
    * 0 < delta <= 1/2, since we require lglg(1/delta) >= 0 (lg = log base 2).

  decompose - if True, returns two matrices X, Y such that XY = AB, though
              X and Y don't take up as much space as A and B.
  '''
  assert epsilon > 0, f'epsilon must be greater than 0. Found epsilon={epsilon}'
  assert 0 <= delta <= 1/2., f'delta must be in (0, 1/2]. Found delta={delta}'
  m, n = A.shape
  nn, p = B.shape
  assert n == nn, f'Dimension mismatch: cannot multiply {(m, n)} by {(nn, p)}.'

  # Step 1, Lines 1-2:
  # Compute lg(1/delta) tug-of-war matrices, S_i, using standard normal.
  N = norm(0, 1)
  beta = np.log2(1/delta)
  S = epsilon * N.rvs(size=discretize((beta, 1/epsilon**2, n)))

  # Step 2, Lines 3-4:
  # Compute lg(1/delta) x 2(lg(1/delta) + lg(lg(1/delta))) tug-of-war matrices,
  # Q_{i,j}, using standard normal.
  gamma = 2*(beta + np.log2(beta))
  # If 1/epsilon^2 = 16, then epsilon = 1/4.
  Q = (1/4.) * N.rvs(size=discretize((beta, gamma, 16, p)))

  # Step 3a: Compute transpose of matrices first.
  St = np.transpose(S, axes=(0, 2, 1))
  Qt = np.transpose(Q, axes=(0, 1, 3, 2))

  # Step 3, Line 5:
  # Compute SB and AS^t for all S.
  SB, ASt = S@B, A@St

  # Step 4, Line 6:
  # Compute A(BQ^t) and then X = A(BQ^t) for all Q.
  BQt = B@Qt
  ABQt = A@BQt
  X = ABQt

  # Step 5, Line 7:
  # Compute (SB)Q^t and then Xhat = (AS^t)(SBQ^t).
  SBQt = np.einsum('hjk,hikl->hijl', SB, Qt)
  Xhat = np.einsum('hjk,hikl->hijl', ASt, SBQt)

  # Step 6, Line 8:
  # Compute y_{i,j} = ||X_{i,j}-Xhat{i,j}||_F^2.
  y = np.linalg.norm(X - Xhat, axis=(2, 3)) ** 2

  # Step 7, Line 9:
  # Compute z_i = median of y_{i,j} over j.
  z = np.median(y, axis=1)

  # Step 8, Line 10:
  # Compute i^* = argmin z_i.
  i = np.argmin(z)

  return (ASt[i], SB[i]) if decompose else ASt[i]@SB[i]
