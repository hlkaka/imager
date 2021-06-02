import torch
import pytorch_lightning as pl
from torch.autograd import grad

# From https://github.com/google-research/fast-soft-sort with modifications to PyTorch
# Not used. Using module directly


def _isotonic_l2(y, sol, device):
    """Solves an isotonic regression problem using PAV.
    Formally, it solves argmin_{v_1 >= ... >= v_n} 0.5 ||v - y||^2.
    Args:
        y: input to isotonic regression, a 1d-array.
        sol: where to write the solution, an array of the same size as y.
    """
    n = y.shape[0]
    target = torch.arange(n, device=device)
    c = torch.ones(n, device=device)
    sums = torch.zeros(n, device=device)

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
    _isotonic_l2(input_s - input_w, solution, device)
    return solution

def _inv_permutation(permutation, device):
    """Returns inverse permutation of 'permutation'."""
    inv_permutation = torch.zeros(len(permutation), dtype=torch.long, device=device)
    inv_permutation[permutation] = torch.arange(len(permutation), device=device)
    return inv_permutation

def isotonic_vjp(vector, solution_):
    start = 0
    return_value = torch.zeros_like(solution_)
    for size in _partition(solution_):
        end = start + size
        #if self.regularization == "l2":
        val = 1. / size # we are only doing l2 reg
        #else:
        #    val = special.softmax(self.input_s[start:end]) # this is for KL
        return_value[start:end] = val * torch.sum(vector[start:end])
        start = end
    return return_value

def _partition(solution, eps=1e-9):
    """Returns partition corresponding to solution."""
    # pylint: disable=g-explicit-length-test
    if len(solution) == 0:
        return []

    sizes = [1]

    for i in range(1, len(solution)):
        if torch.abs(solution[i] - solution[i - 1]) > eps:
            sizes.append(0)
        sizes[-1] += 1

    return sizes

class Projection():
    def compute_tensor(self, input_theta, input_w=None, device=None):
        if device is None:
            device = input_theta.device

        self.size = input_theta.shape[1]

        single_projection = lambda x : self.compute_single(x, device, input_w=input_w)

        projections_, permutations_, inv_permutations_, iso_sols_ = [], [], [], []

        for batch in input_theta:
            prjcn, perm, inv_prm, iso_sol = single_projection(batch)
            projections_.append(prjcn)
            permutations_.append(perm)
            inv_permutations_.append(inv_prm)
            iso_sols_.append(iso_sol)
        
        projections = torch.stack(projections_)
        self.permutations = torch.stack(permutations_)
        self.inv_permutations = torch.stack(inv_permutations_)
        self.iso_sols = torch.stack(iso_sols_)

        return projections

    def compute_single(self, input_theta, device, input_w=None):
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
        return primal_sol[inv_permutation], permutation, inv_permutation, isotonic_

    def vjp_single(self, vector, sol, perm, inv_perm):
        ret = vector.clone()
        ret -= isotonic_vjp(vector[perm], sol)[inv_perm]
        return ret

    def vjp(self, vector):
        rets = []
        for v, s, p, ip in zip(vector, self.iso_sols, self.permutations, self.inv_permutations):
            rets.append(self.vjp_single(v, s, p, ip))
        return torch.stack(rets)

class SR(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        proj = Projection()
        ctx.proj = proj
        return proj.compute_tensor(input)

    @staticmethod
    def backward(ctx, grad_output):
        return ctx.proj.vjp(grad_output)

class SoftRank(pl.LightningModule):
    '''
    Calculates the soft rank of a given sequence
    Differentiable
    '''
    
    def __init__(self, direction="ASCENDING",
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
        self.sr = SR.apply

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
            return self.sr(input_theta * self.scale)


if __name__ == '__main__':
    #t = torch.tensor([3, 2, 6, 2])
    #t = torch.tensor([5, 10, 12, 1])
    t = torch.tensor([[2.0, 7, 5, 4],[2, 4, 17, 3]])
    gt = torch.tensor([[1, 4, 3, 2], [1, 2, 3, 4]])

    print(t)

    sr = SoftRank()

    answer = sr(t)

    print(answer)

    loss = torch.nn.MSELoss(reduction='mean')
    print("Loss: {}".format(loss(gt, answer)))