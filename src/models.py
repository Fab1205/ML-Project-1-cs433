import numpy as np
from implementations import (
    batch_iter,
    compute_gradient_logistic,
    compute_gradient_mse,
    compute_loss_logistic,
    compute_loss_mse,
    ridge_regression,
    sigmoid,
)

def least_squares(y: np.array, tx: np.array):
    """
    Compute the least squares solution, falling back to pseudo-inverse if necessary.

    Args:
        y (np.array): labels
        tx (np.array): features

    Returns:
        np.array: the weights
        float: the loss
    """
    try:
        # Attempt to solve using np.linalg.solve
        w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    except np.linalg.LinAlgError:
        # Fall back to pseudo-inverse in case of singular matrix
        print("Matrix is singular; using pseudo-inverse.")
        w = np.linalg.pinv(tx.T.dot(tx)).dot(tx.T).dot(y)

    # Compute the loss
    loss = compute_loss_mse(y, tx, w)
    return w, loss


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

    losses = []
    weights = []

    for n_iter in range(max_iters):
        gradient = compute_gradient_mse(y, tx, w)
        w = w - gamma * gradient

        loss = compute_loss_mse(y, tx, w)
        losses.append(loss)
        weights.append(w.copy())

    return w, losses, weights


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

    losses = []
    weights = []

    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            gradient = compute_gradient_mse(minibatch_y, minibatch_tx, w)
            w = w - gamma * gradient

        loss = compute_loss_mse(y, tx, w)
        losses.append(loss)
        weights.append(w.copy())

    return w, losses, weights

def logistic_regression(
    y: np.ndarray,
    tx: np.ndarray,
    initial_w: np.ndarray,
    max_iters: int,
    gamma: float,
    class_weight=1,
    y_val=None,
    tx_val=None,
    patience=10,
):
    """
    Logistic regression using gradient descent with early stopping.

    Args:
        y (np.array): Training labels
        tx (np.array): Training features
        initial_w (np.array): Initial weights
        max_iters (int): Maximum number of iterations
        gamma (float): Learning rate
        class_weight (float): Weight for the positive class (default: 1)
        y_val (np.array, optional): Validation labels
        tx_val (np.array, optional): Validation features
        patience (int, optional): Number of iterations with no improvement before stopping

    Returns:
        w (np.array): Final weights
        losses (list): List of training losses at each iteration
        weights (list): List of weight updates at each iteration
    """
    w = initial_w
    losses = []
    weights = []

    best_val_loss = np.inf  # Initialize best validation loss as infinity
    no_improvement_count = 0  # Counter for early stopping

    for n_iter in range(max_iters):
        # Compute prediction and error
        pred = sigmoid(tx.dot(w))
        error = pred - y

        # Adjust for class imbalance with class weight
        weighted_error = np.where(y == 1, class_weight * error, error)

        gradient = tx.T.dot(weighted_error) / len(y)
        w = w - gamma * gradient

        train_loss = compute_loss_logistic(y, tx, w)
        losses.append(train_loss)
        weights.append(w.copy())

        if y_val is not None and tx_val is not None:
            val_loss = compute_loss_logistic(y_val, tx_val, w)

        else:
            val_loss = train_loss  # Use training loss if no validation set

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # Stop if no improvement over 'patience' iterations
        if no_improvement_count >= patience:
            print(f"Early stopping at iteration {n_iter} due to no improvement.")
            break

    return w, losses, weights


def logistic_regression(
    y: np.ndarray,
    tx: np.ndarray,
    initial_w: np.ndarray,
    max_iters: int,
    gamma: float,
    class_weight=1,
    y_val=None,
    tx_val=None,
    patience=10,
):
    """
    Logistic regression using gradient descent with early stopping.

    Args:
        y (np.array): Training labels
        tx (np.array): Training features
        initial_w (np.array): Initial weights
        max_iters (int): Maximum number of iterations
        gamma (float): Learning rate
        class_weight (float): Weight for the positive class (default: 1)
        y_val (np.array, optional): Validation labels
        tx_val (np.array, optional): Validation features
        patience (int, optional): Number of iterations with no improvement before stopping

    Returns:
        w (np.array): Final weights
        losses (list): List of training losses at each iteration
        weights (list): List of weight updates at each iteration
    """
    w = initial_w
    losses = []
    weights = []
    best_val_loss = np.inf
    no_improvement_count = 0

    for n_iter in range(max_iters):

        pred = sigmoid(tx.dot(w))
        error = pred - y

        # Adjust for class imbalance with class weight
        weighted_error = np.where(y == 1, class_weight * error, error)

        gradient = tx.T.dot(weighted_error) / len(y)
        w = w - gamma * gradient

        train_loss = compute_loss_logistic(y, tx, w)
        losses.append(train_loss)
        weights.append(w.copy())

        if y_val is not None and tx_val is not None:
            val_loss = compute_loss_logistic(y_val, tx_val, w)

        else:
            val_loss = train_loss  # Use training loss if no validation set

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # Stop if no improvement over 'patience' iterations
        if no_improvement_count >= patience:
            print(f"Early stopping at iteration {n_iter} due to no improvement.")
            break

    return w, losses, weights


def reg_l2_logistic_regression(
    y: np.array, tx: np.array, lambda_: float, initial_w: np.array, max_iters: int, gamma: float, class_weight=1
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

    losses = []
    weights = []

    for n_iter in range(max_iters):
        pred = 1 / (1 + np.exp(-tx.dot(w)))  # Sigmoid prediction
        error = pred - y

        # Adjust for class imbalance with class weight
        weighted_error = np.where(y == 1, class_weight * error, error)

        gradient = tx.T.dot(weighted_error) / len(y) + 2 * lambda_ * w
        w = w - gamma * gradient

        loss = compute_loss_logistic(y, tx, w)
        losses.append(loss)
        weights.append(w.copy())

    return w, losses, weights


def reg_l1_logistic_regression(
    y: np.array, tx: np.array, lambda_: float, initial_w: np.array, max_iters: int, gamma: float, class_weight=1
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

    losses = []
    weights = []

    for n_iter in range(max_iters):
        pred = 1 / (1 + np.exp(-tx.dot(w)))
        error = pred - y

        # Adjust for class imbalance with class weight
        weighted_error = np.where(y == 1, class_weight * error, error)

        gradient = tx.T.dot(weighted_error) / len(y) + (lambda_ * np.sign(w))
        w = w - gamma * gradient

        loss = compute_loss_logistic(y, tx, w)
        losses.append(loss)
        weights.append(w.copy())

    return w, losses, weights
