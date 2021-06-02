import unittest
from absl.testing import parameterized

import numpy as np
from sklearn.isotonic import isotonic_regression
import torch.autograd as autograd
import torch

import sys
sys.path.append('.')
from models.soft_rank import _isotonic_l2
from fast_soft_sort.pytorch_ops import soft_rank, wrap_class
from fast_soft_sort import numpy_ops
from models.soft_rank import SoftRank

def isotonic_l2(y, sol):
  """Solves an isotonic regression problem using PAV.
  Formally, it solves argmin_{v_1 >= ... >= v_n} 0.5 ||v - y||^2.
  Args:
    y: input to isotonic regression, a 1d-array.
    sol: where to write the solution, an array of the same size as y.
  """
  n = y.shape[0]
  target = np.arange(n)
  c = np.ones(n)
  sums = np.zeros(n)

  # target describes a list of blocks.  At any time, if [i..j] (inclusive) is
  # an active block, then target[i] := j and target[j] := i.

  for i in range(n):
    sol[i] = y[i]
    sums[i] = y[i]

  i = 0
  while i < n:
    k = target[i] + 1
    if k == n:
      break
    if sol[i] > sol[k]:
      i = k
      continue
    sum_y = sums[i]
    sum_c = c[i]
    while True:
      # We are within an increasing subsequence.
      prev_y = sol[k]
      sum_y += sums[k]
      sum_c += c[k]
      k = target[k] + 1
      if k == n or prev_y > sol[k]:
        # Non-singleton increasing subsequence is finished,
        # update first entry.
        sol[i] = sum_y / sum_c
        sums[i] = sum_y
        c[i] = sum_c
        target[i] = k - 1
        target[k - 1] = i
        if i > 0:
          # Backtrack if we can.  This makes the algorithm
          # single-pass and ensures O(n) complexity.
          i = target[i - 1]
        # Otherwise, restart from the same point.
        break

  # Reconstruct the solution.
  i = 0
  while i < n:
    k = target[i] + 1
    sol[i + 1 : k] = sol[i]
    i = k



class IsotonicTest(parameterized.TestCase):

  def test_l2_agrees_with_sklearn(self):
    rng = np.random.RandomState(0)
    y = rng.randn(10) * rng.randint(1, 5)
    sol = torch.zeros_like(torch.tensor(y, device='cpu'))
    _isotonic_l2(torch.tensor(y), sol, 'cpu')
    sol_pkg = np.zeros_like(y)
    isotonic_l2(y, sol_pkg)
    sol_skl = isotonic_regression(y, increasing=False)
    np.testing.assert_array_almost_equal(sol_pkg, sol_skl)
    np.testing.assert_array_almost_equal(sol.detach().numpy(), sol_pkg)

class BackwardTest(parameterized.TestCase):

  def test_backward_agrees(self):
    t = autograd.Variable(torch.tensor([[4., 34., 52., 3., 2.], [5., 3., 66., 3., 32.]]), requires_grad=True)
    sr = SoftRank()

    gt = torch.tensor([[3., 4., 5., 2., 1.], [3., 1., 5., 2., 4.]])

    loss = torch.nn.MSELoss(reduction='mean')

    sol_sr = sr(t)
    sol_pkg = soft_rank(t)

    l_sr = loss(sol_sr, gt)
    l_sr.backward()
    print(t.grad)

    t.grad.data.zero_()

    l_pkg = loss(sol_pkg, gt)
    l_pkg.backward()
    print(t.grad)

if __name__ == "__main__":
  unittest.main()