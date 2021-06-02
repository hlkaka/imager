import torch
import pytorch_lightning as pl

# From https://github.com/google-research/fast-soft-sort with modifications to PyTorch

def _isotonic_l2(y, sol, device):
    """Solves an isotonic regression problem using PAV.
    Formally, it solves argmin_{v_1 >= ... >= v_n} 0.5 ||v - y||^2.
    Args:
        y: input to isotonic regression, a 1d-array.
        sol: where to write the solution, an array of the same size as y.
    """
    n = y.shape[0]
    target = torch.arange(n, device=device)
    c = torch.arange(n, device=device)
    sums = torch.arange(n, device=device)

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

def isotonic_l2(input_s, device, input_w=None):
    """Solves an isotonic regression problem using PAV.
    Formally, it solves argmin_{v_1 >= ... >= v_n} 0.5 ||v - (s-w)||^2.
    Args:
    input_s: input to isotonic regression, a 1d-array.
    input_w: input to isotonic regression, a 1d-array.
    Returns:
    solution to the optimization problem.
    """
    if input_w is None:
        theta_l = torch.arange(len(input_s), device=device)
        input_w = torch.flip(theta_l, [0]) + 1

    input_w = input_w.type(input_s.dtype)
    solution = torch.zeros_like(input_s, device=device)
    _isotonic_l2(input_s - input_w, solution, device=device)
    return solution

def _inv_permutation(permutation, device):
    """Returns inverse permutation of 'permutation'."""
    inv_permutation = torch.zeros(len(permutation), dtype=torch.long, device=device)
    inv_permutation[permutation] = torch.arange(len(permutation), device=device)
    return inv_permutation

def projection(input_theta, input_w=None, device='cpu'):
    # From constructor
    # Only L2 regularization is implemented
    # regularization = 'l2'
    
    if input_w is None:
        theta_l = torch.arange(len(input_theta), device=device)
        input_w = torch.flip(theta_l, [0]) + 1

    # From compute
    permutation = torch.flip(torch.argsort(input_theta), [0])
    input_s = input_theta[permutation]

    isotonic_ = isotonic_l2(input_s, device=device, input_w=input_w) #, regularization=regularization)
    dual_sol = isotonic_
    primal_sol = input_s - dual_sol

    inv_permutation = _inv_permutation(permutation, device=device)
    return primal_sol[inv_permutation]

class SoftRank(pl.LightningModule):
    '''
    Calculates the soft rank of a given sequence
    Differentiable
    '''
    
    def __init__(self, length :int, direction="ASCENDING",
               regularization_strength=1.0, regularization="l2"):
        if direction not in ("ASCENDING", "DESCENDING"):
            raise ValueError("direction should be either 'ASCENDING' or 'DESCENDING'")

        # 'kl' regularization is not implemented
        if regularization not in ("l2"):
            raise ValueError("'regularization' should be either 'l2' or 'kl' "
                     "but got %s. [Note 'kl' is not implemented yet]" % str(regularization))

        super().__init__()

        sign = 1 if direction == "ASCENDING" else -1
        self.scale = sign / regularization_strength

        self.regularization = regularization
        l_ar = torch.arange(length)
        input_w = torch.flip(l_ar, [0]) + 1 # torch.arange(length)[::-1] + 1
        self.register_buffer('input_w', input_w)

    def forward(self, input_theta):
        if self.regularization == "kl":
            raise NotImplementedError('KL regularization is not yet implemented')
            #self.projection_ = Projection(
            #    self.values * self.scale,
            #    np.log(self.input_w),
            #    regularization=self.regularization)
            #self.factor = np.exp(self.projection_.compute())
            #return self.factor
        else:
            project = lambda x : projection(x * self.scale, input_w=self.input_w, device=self.device)
            return torch.stack([project(batch) for batch in input_theta])



if __name__ == '__main__':
    sr = SoftRank(4)

    #t = torch.tensor([3, 2, 6, 2])
    #t = torch.tensor([5, 10, 12, 1])
    t = torch.tensor([[2.0, 7, 5, 4],[2, 4, 8, 3]])
    gt = torch.tensor([[1, 4, 3, 2], [1, 2, 3, 4]])

    print(t)

    answer = torch.stack([sr(row) for row in t])

    print(answer)

    loss = torch.nn.MSELoss(reduction='mean')
    print("Loss: {}".format(loss(gt, answer)))