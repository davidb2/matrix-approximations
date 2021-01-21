#!/usr/bin/env python3
import numpy as np

from scipy.stats.distributions import norm

def discretize(xs):
  '''Converts a float tuple into an int tuple by taking the ceiling.'''
  return tuple(np.int32(np.ceil(x)) for x in xs)


def regression(A, b, epsilon=0.25, delta=0.25, loss=False):
  '''
  l2 regression (Theorem 12)
  without the use of the Fast Johnson-Lindenstrauss Transform (FJLT).

  Let x^* = argmin_x ||b-Ax||_2 and Z = ||b-Ax^*||_2 be the smallest l2 loss.
  Let xp be the output of this function and Zp be its l2 loss under JLT.
  Then:
    (*) Pr(Zp <= (1+epsilon)*Z) >= 1-delta,
    (*) Pr(||b-A(xp)||_2 <= (1+epsilon)*Z) >= 1-delta,
    (*) Pr(||x^*-xp||_2 <= (epsilon/sigma)*Z) >= 1-delta,
    where sigma is the smallest singular value of A.

  Time complexity of:
    O(brd*(n+r))
    * b = beta = O(lg(1/delta))
    * r = (d*log d)/epsilon^2
  with
    * 0 < epsilon < 1
    * 0 < delta < 1

  loss - if True, returns xp along with its regular and JLT l2 loss as a tuple;
         otherwise, just returns xp.

  '''
  assert 0 < epsilon < 1, f'epsilon must be in (0, 1). Found epsilon={epsilon}'
  assert 0 < delta < 1, f'delta must be in (0, 1). Found delta={delta}'
  n, d = A.shape
  b = np.atleast_2d(b).T
  nn, q = b.shape
  assert q == 1, f'b should be of shape (n, 1) or (n,).'
  assert n == nn, f'Dimension mismatch: cannot multiply {(n, d)} by {(nn, 1)}.'

  # Set r = \Omega(e^{-2) * d * log d) for (2-4) to hold in Theorem 12.
  r = d*np.log(d)/epsilon**2

  # Set a JLT from R^n to R^r.
  N = norm(0, 1)
  beta = np.log(1/delta)
  S = (1/np.sqrt(r)) * N.rvs(size=discretize((beta, r, n)))

  # Compute SA, Sb, (SA)^+, and x_opt = (SA)^+SB for each copy of S.
  SA, Sb = S@A, S@b
  SAp = np.linalg.pinv(SA)
  xps = SAp@Sb

  # Find the best x.
  Z = np.linalg.norm(Sb - SA@xps, axis=(1, 2))
  i = np.argmin(Z)
  xp = xps[i]

  return (xp, np.linalg.norm(b-A@xp), Z[i]) if loss else xp
