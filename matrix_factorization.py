# (C) Kyle Kastner, June 2014
# License: BSD 3 clause
# Latest version can be found at:
# https://gist.github.com/kastnerkyle/9341182

import numpy as np
from scipy import sparse


def minibatch_indices(X, minibatch_size):
    minibatch_indices = np.arange(0, len(X), minibatch_size)
    minibatch_indices = np.asarray(list(minibatch_indices) + [len(X)])
    start_indices = minibatch_indices[:-1]
    end_indices = minibatch_indices[1:]
    return zip(start_indices, end_indices)


def shuffle_in_unison(a, b):
    """
    http://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
    """
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)


def PMF(X, rank=10, learning_rate=0.001, momentum=0.8,
        regularization=0.25, minibatch_size=1000, max_epoch=1000,
        nan_value=0, status_percentage=0.1, random_state=None):
    """
    Python implementation of Probabilistic Matrix Factorization (PMF).

    Parameters
    ----------
    X: numpy array or scipy.sparse coo matrix, shape (n_users, n_items)
        Input data. If a dense array is passed in, it will be converted to a
        sparse matrix by looking for all `nan_value` numbers and treating them
        as empty.

    rank: int, optional (default=10)
       Rank of the low-rank factor matrices. A higher rank should result in a
       better approximation, at the cost of more memory and slower computataion.

    learning_rate: float, optional (default=0.001)
        Learning rate for minibatch gradient descent.

    momentum: float, optional (default=0.8)
        Momentum for minibatch gradient descent.

    regularization: float, optional (default=0.25)
        L2 regularization penalty for minibatch gradient descent.

    minibatch_size: int, optional (default=1000)
       The size of each minibatch. If this is larger than size of the dataset,
       will default to running over the whole dataset.

    max_epoch: int, optional (default=1000)
        The maximum number of epochs.

    nan_value: int, optional (default=0)
        This value will be masked out of the input for calculations
        Should match the value considered the "not rated" in the dataset X.

    status_percentage: float in (0, 1), optional (default=0.1)
        The relative percentage of `max_epochs` when status will be printed.
        For example, 0.1 is every 10%, 0.01 is every 1%, and so on. For
        the default values of max_epoch=1000, status_percentage=0.1 this
        is equivalent to a status print every 100 epochs.

    random_state: RandomState, int, or None, optional (default=None)
        Random state to pass in. Can be an int, None, or np.random.RandomState
        object.


    Returns
    -------
    U: array-like, shape (X.shape[0], rank)
        Row basis for reconstruction.
        Usage:
        reconstruction = np.dot(U, V.T) + X_mean

    V: array-like, shape (X.shape[1], rank)
        Column basis for reconstruction.
        Usage:
        reconstruction = np.dot(U, V.T) + X_mean

    X_mean: float
        Global mean prediction, needed for reconstruction
        Usage
        reconstruction = np.dot(U, V.T) + X_mean


    Notes
    -----
    Based on code from Ruslan Salakhutdinov
    http://www.cs.toronto.edu/~rsalakhu/code_BPMF/pmf.m

    Probabilistic Matrix Factorization, R. Salakhutdinov and A. Mnih,
    Advances in Neural Information Processing Systems 20, 2008
    """
    if not sparse.isspmatrix_coo(X):
        val_index = np.where(X != nan_value)
        X = sparse.coo_matrix((X[val_index[0], val_index[1]],
                               (val_index[0], val_index[1])))
    # Simplest prediction is the global mean
    X_mean = X.mean()
    lr = learning_rate
    reg = regularization
    mom = momentum
    if random_state is None or type(random_state) is int:
        random_state = np.random.RandomState(random_state)
    N, M = X.shape
    U = 0.1 * random_state.randn(N, rank)
    V = 0.1 * random_state.randn(M, rank)
    U_inc = np.zeros_like(U)
    V_inc = np.zeros_like(V)
    dU = np.zeros_like(U)
    dV = np.zeros_like(V)
    epoch = 0
    status_inc = int(np.ceil(max_epoch * status_percentage))
    print("Printing updates every %i epochs" % status_inc)
    status_points = list(range(0, max_epoch, status_inc)) + [max_epoch - 1]
    # Need this in order to index
    X_s = X.tolil()
    while epoch < max_epoch:
        # Get indices for non-NaN values
        r, c = X.nonzero()
        mb_indices = minibatch_indices(zip(r, c), minibatch_size)
        n_batches = len(mb_indices)
        shuffle_in_unison(r, c)
        mean_abs_err = 0.
        for i, j in mb_indices:
            # Reset derivative matrices each minibatch
            dU[:, :] = 0.
            dV[:, :] = 0.
            # Slice out row and column indices
            r_i = r[i:j]
            c_i = c[i:j]
            # Get data corresponding to the row and column indices
            X_i = X_s[r_i, c_i].toarray().ravel() - X_mean
            # Compute predictions
            pred = np.sum(U[r_i] * V[c_i], axis=1)
            # Compute how algorithm is doing
            mean_abs_err += np.sum(np.abs(pred - X_i)) / (n_batches * (j - i))
            # Loss has a tendency to be unstable, but is the "right thing"
            # to monitor instead of sum_abs_err
            # pred_loss = (pred - X_i) ** 2
            # Compute gradients
            grad_loss = 2 * (pred - X_i)
            grad_U = grad_loss[:, None] * V[c_i] + reg * U[r_i]
            grad_V = grad_loss[:, None] * U[r_i] + reg * V[c_i]
            dU[r_i] = grad_U
            dV[c_i] = grad_V
            # Momentum storage
            U_inc = mom * U_inc + lr * dU
            V_inc = mom * V_inc + lr * dV
            U -= U_inc
            V -= V_inc
        if epoch in status_points:
            print("Epoch %i of %i" % (epoch + 1, max_epoch))
            print("Mean absolute error %f" % (mean_abs_err))
        epoch += 1
    return U, V, X_mean


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    R = np.array([[5, 3, 0, 1],
                  [4, 0, 0, 1],
                  [1, 1, 0, 5],
                  [1, 0, 0, 4],
                  [0, 1, 5, 4]], dtype=float)
    U, V, m = PMF(R, learning_rate=0.001, momentum=0.95,
                  minibatch_size=2, rank=5, max_epoch=250, random_state=1999)
    R2 = np.dot(U, V.T) + m
    plt.matshow(R * (R > 0))
    plt.title("Ground truth ratings")
    plt.matshow(R2 * (R > 0))
    plt.title("Predicted ratings")
    plt.show()
