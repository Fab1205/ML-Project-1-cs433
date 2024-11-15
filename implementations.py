import numpy as np


def sigmoid(t: np.array) -> np.array:
    """
    Sigmoid function.

    Args:
        t (np.array): input to the sigmoid function
    Returns:
        np.array: output of the sigmoid function
    """

    return 1 / (1 + np.exp(-t))


def compute_loss_mse(y: np.array, tx: np.array, w: np.array) -> float:
    """
    Compute the loss of the mean squared error.

    Args:
        y (np.array): labels
        tx (np.array): features
        w (np.array): weights
    Returns:
        float: the loss of the mean squared error
    """
    e = y - tx.dot(w)
    return 1 / 2 * np.mean(e**2)


def compute_gradient_mse(y: np.array, tx: np.array, w: np.array) -> np.array:
    """
    Compute the gradient of the mean squared error.

    Args:
        y (np.array): labels
        tx (np.array): features
        w (np.array): weights
    Returns:
        np.array: the gradient of the mean squared error
    """
    e = y - tx.dot(w)
    return -tx.T.dot(e) / len(e)


def compute_loss_logistic(y: np.array, tx: np.array, w: np.array) -> float:
    """
    Compute the loss for logistic regression when labels are 0 and 1.

    Args:
        y (np.array): labels (binary: 0 or 1, or -1 and 1)
        tx (np.array): features
        w (np.array): weights
    Returns:
        float: the loss of the logistic regression
    """
    # Ensure y is in {0, 1} form
    y_ = np.where(y == -1, 0, y)

    pred = sigmoid(np.dot(tx, w))
    loss = -np.mean(
        y_ * np.log(pred) + (1 - y_) * np.log(1 - pred)
    )  # add epsilon to avoid log(0) # but need to redo tests
    return loss


def compute_gradient_logistic(y: np.array, tx: np.array, w: np.array) -> np.array:
    """
    Compute the gradient of the logistic regression.

    Args:
        y (np.array): labels
        tx (np.array): features
        w (np.array): weights
    Returns:
        np.array: the gradient of the logistic regression
    """
    # Ensure y is in {0, 1} form
    y_ = np.where(y == -1, 0, y)

    pred = sigmoid(tx.dot(w))
    gradient = tx.T.dot(pred - y_) / len(y_)
    return gradient


def batch_iter(y: np.array, tx: np.array, batch_size, num_batches=1, shuffle=True):
    """

    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.

    Example:

     Number of batches = 9

     Batch size = 7                              Remainder = 3
     v     v                                         v v
    |-------|-------|-------|-------|-------|-------|---|
        0       7       14      21      28      35   max batches = 6

    If shuffle is False, the returned batches are the ones started from the indexes:
    0, 7, 14, 21, 28, 35, 0, 7, 14

    If shuffle is True, the returned batches start in:
    7, 28, 14, 35, 14, 0, 21, 28, 7

    To prevent the remainder datapoints from ever being taken into account, each of the shuffled indexes is added a random amount
    8, 28, 16, 38, 14, 0, 22, 28, 9

    This way batches might overlap, but the returned batches are slightly more representative.

    Disclaimer: To keep this function simple, individual datapoints are not shuffled. For a more random result consider using a batch_size of 1.

    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>

    this is a helper function given in the labs
    """
    data_size = len(y)  # NUmber of data points.
    batch_size = min(data_size, batch_size)  # Limit the possible size of the batch.
    max_batches = int(
        data_size / batch_size
    )  # The maximum amount of non-overlapping batches that can be extracted from the data.
    remainder = data_size - max_batches * batch_size  # Points that would be excluded if no overlap is allowed.

    if shuffle:
        # Generate an array of indexes indicating the start of each batch
        idxs = np.random.randint(max_batches, size=num_batches) * batch_size
        if remainder != 0:
            # Add an random offset to the start of each batch to eventually consider the remainder points
            idxs += np.random.randint(remainder + 1, size=num_batches)
    else:
        # If no shuffle is done, the array of indexes is circular.
        idxs = np.array([i % max_batches for i in range(num_batches)]) * batch_size

    for start in idxs:
        start_index = start  # The first data point of the batch
        end_index = start_index + batch_size  # The first data point of the following batch
        yield y[start_index:end_index], tx[start_index:end_index]


def mean_squared_error_gd(y: np.array, tx: np.array, initial_w: np.array, max_iters: int, gamma: float):
    """
    Compute the mean squared error using gradient descent.

    Args:
        y (np.array): labels
        tx (np.array): features
        initial_w (np.array): initial weights
        max_iters (int): maximum number of iterations
        gamma (float): learning rate
    Returns:
        np.array: the weights
        float: the loss
    """
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient_mse(y, tx, w)
        w = w - gamma * gradient
    return w, compute_loss_mse(y, tx, w)


def mean_squared_error_sgd(y: np.array, tx: np.array, initial_w: np.array, max_iters: int, gamma: float, batch_size=1):
    """
    Compute the mean squared error using stochastic gradient descent.

    Args:
        y (np.array): labels
        tx (np.array): features
        initial_w (np.array): initial weights
        max_iters (int): maximum number of iterations
        gamma (float): learning rate
        batch_size (int): size of the batch
    Returns:
        np.array: the weights
        float: the loss
    """
    w = initial_w
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            gradient = compute_gradient_mse(minibatch_y, minibatch_tx, w)
            w = w - gamma * gradient
    return w, compute_loss_mse(y, tx, w)


def least_squares(y: np.array, tx: np.array):
    """
    Compute the least squares.

    Args:
        y (np.array): labels
        tx (np.array): features
    Returns:
        np.array: the weights
        float: the loss
    """
    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    return w, compute_loss_mse(y, tx, w)


def ridge_regression(y: np.array, tx: np.array, lambda_: float):
    """
    Compute the ridge regression.

    Args:
        y (np.array): labels
        tx (np.array): features
        lambda_ (float): regularization parameter
    Returns:
        np.array: the weights
        float: the loss
    """
    N = tx.shape[0]
    D = tx.shape[1]
    w = np.linalg.solve(tx.T.dot(tx) + 2 * N * lambda_ * np.eye(D), tx.T.dot(y))
    return w, compute_loss_mse(y, tx, w)


def logistic_regression(y: np.array, tx: np.array, initial_w: np.array, max_iters: int, gamma: float):
    """
    Compute the logistic regression.

    Args:
        y (np.array): labels
        tx (np.array): features
        initial_w (np.array): initial weights
        max_iters (int): maximum number of iterations
        gamma (float): learning rate
    Returns:
        np.array: the weights
        float: the loss
    """
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient_logistic(y, tx, w)
        w = w - gamma * gradient
        loss = compute_loss_logistic(y, tx, w)
    return w, loss


def reg_logistic_regression(
    y: np.array, tx: np.array, lambda_: float, initial_w: np.array, max_iters: int, gamma: float
):
    """
    Compute the regularized logistic regression.

    Args:
        y (np.array): labels
        tx (np.array): features
        lambda_ (float): regularization parameter
        initial_w (np.array): initial weights
        max_iters (int): maximum number of iterations
        gamma (float): learning rate
    Returns:
        np.array: the weights
        float: the loss
    """
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient_logistic(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * gradient
    return w, compute_loss_logistic(y, tx, w)
