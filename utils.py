import numpy as np
import matplotlib.pyplot as plt
import torch


def log_sum_exp(tensor, dim=-1, sum_op=torch.sum):
    """
    Uses the LogSumExp (LSE) as an approximation for the sum in a log-domain.
    :param tensor: Tensor to compute LSE over
    :param dim: dimension to perform operation over
    :param sum_op: reductive operation to be applied, e.g. torch.sum or torch.mean
    :return: LSE
    """
    max, _ = torch.max(tensor, dim=dim, keepdim=True)
    return torch.log(sum_op(torch.exp(tensor - max), dim=dim, keepdim=True) + 1e-8) + max


def flat_to_triangular(flat_params, with_diagonal=False):
    L = len(flat_params)
    N = int((-1 + np.sqrt(1 + 8 * L)) // 2)   # matrix size from num params: L = N(N+1)/2
    if not with_diagonal:
        N += 1
    A = torch.zeros((N,N),device=flat_params.device, dtype=flat_params.dtype)
    k=0
    if with_diagonal:
        for i in range(N):
            A[i, :i+1] = flat_params[k:k+i+1]
            k = k+i+1
    else:
        for i in range(1, N):
            A[i, :i] = flat_params[k:k+i]
            k = k+i
    return A

def triangular_to_flat(A):   # we don't need this, but if we do we have to implement with_diagonal=False
    N = A.size(0)
    L = (N*(N+1))//2
    flat_params = torch.zeros((L), device=A.device, dtype=A.dtype)
    k=0
    for i in range(N):
        flat_params[k:k+i+1] = A[i, :i+1] 
        k = k+i+1
    return flat_params



def make_cholesky(logvar_ab, cov_ab):
    m = torch.diag(torch.ones_like(logvar_ab))
    std = (logvar_ab/2).exp()
    return (1-m) * cov_ab + torch.diag(std)


def onehot(a):
    """
    Get one-hot representation of tensor of any number of dimensions.
    Append one-hot dimension at the end.
    """
    
    ncols = a.max() + 1
    out = np.zeros((a.size, ncols), dtype=np.uint8)
    out[np.arange(a.size), a.ravel()] = 1
    out = out.reshape(a.shape + (ncols,))
    return out


def plot(x=None, y=None, y_lower=None, y_upper=None, color=None,
         alpha=0.2, line_alpha=1.0, title='', xlabel='', ylabel='', label='', **kwargs):
    ax = plt.gca()
    if x is None:
        x = range(len(y))
    assert x.ndim == 1
    ax.plot(x, y, color=color, label=label, alpha=line_alpha, **kwargs)

    if y_lower is not None and y_upper is None:
        y_upper = y
    elif y_lower is None and y_upper is not None:
        y_lower = y
    # Now both lower and upper are either None or not
    if y_lower is not None:
        add_shading(ax, x=x, lower=y_lower, upper=y_upper, color=color, alpha=alpha)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if label != '':
        ax.legend()
    ax.grid(True)

def add_shading(ax, x=None, lower=None, upper=None, color=None, alpha=0.2):
    if x is None:
        x = range(len(lower))
    ax.fill_between(np.array(x), np.array(lower), np.array(upper), color=color, alpha=alpha)

def clean_curr_axes():
    plt.gca().tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        left=False,  # ticks along the left edge are off
        right=False,
        bottom=False,
        top=False,
        labelleft=False,  # labels along the left edge are off
        labelbottom=False)
    
if __name__ == '__main__':
    flat_params = torch.tensor([1,2,3,4,5,6])
    print("Input:\n", flat_params)

    tri = flat_to_triangular(flat_params)
    print("Matrix:\n", tri)

    print("Flatten:\n", triangular_to_flat(tri))
