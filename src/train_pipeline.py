import inspect
import itertools
import json
import os
import time
from collections import defaultdict
from datetime import datetime
from itertools import product
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from utils import calculate_accuracy, calculate_f1_score, predict_labels


def train_and_test(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    model_func: callable,
    metrics="f1",
    verbose=False,
    **hyperparameters,
):
    """
    Train and test a model with the given hyperparameters and print and save the results.

    Args:
        x_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        x_test (np.ndarray): Test features.
        y_test (np.ndarray): Test labels.
        model_func (callable): The model function to train.
        metrics (str): Metric to optimize, defaults to "f1".
        verbose (bool): Whether to print detailed information.
        **hyperparameters: Hyperparameters for the model.

    Returns:
        Tuple containing:
            - Tuple containing the weights of the trained model and the final loss
            - Dictionary containing the evaluation metrics
    """

    def format_training_time(seconds: float) -> str:
        """
        Format the training time in a human-readable way.
        """
        if seconds < 60:
            return f"{seconds:.2f} s"
        elif seconds < 3600:
            minutes = seconds // 60
            remaining_seconds = seconds % 60
            return f"{int(minutes)} min {remaining_seconds:.2f} s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{int(hours)} hr {int(minutes)} min"

    # Inspect model signature and filter parameters
    model_signature = inspect.signature(model_func)
    model_params = model_signature.parameters
    model_specific_params = {k: v for k, v in hyperparameters.items() if k in model_params}

    if verbose:
        # Filter out 'initial_w' from the parameters to be printed
        filtered_params = {k: v for k, v in model_specific_params.items() if k != "initial_w"}

        print("Model specific parameters:")
        for k, v in filtered_params.items():
            print(f"    {k}: {v}")
        print()

    if metrics != "f1":
        raise ValueError("Only f1 metric is supported")

    # Training the model
    start_time = time.time()
    print("Training model...")
    model_output = model_func(y_train, x_train, **model_specific_params)  # Maybe change back to (w, loss) output ?
    w, losses, weights = standardize_model_output(model_output)
    end_time = time.time()
    training_time = end_time - start_time
    formatted_time = format_training_time(training_time)

    last_loss = losses[-1]
    last_weights = weights[-1]

    print(f"Training done in {formatted_time} with last loss: {last_loss:.5f}")

    # Generate unique ID with model name
    model_name = model_func.__name__
    unique_id = f"{model_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    # Directories for saving results
    base_path = "../results"
    log_path = os.path.join(base_path, "logs", unique_id)

    os.makedirs(log_path, exist_ok=True)

    # Save last weights and loss
    results_data = {
        "weights": last_weights.tolist(),
        "loss": last_loss,
    }

    with open(os.path.join(log_path, "results.json"), "w") as f:
        json.dump(results_data, f, indent=4)

    # Plot the loss
    if len(losses) > 1:
        plt.plot(losses, color="blue")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.legend(["Training Loss"])

        if verbose:
            plt.show()

        plt.savefig(os.path.join(log_path, f"{unique_id}_loss_plot.png"))
        plt.close()

    # Evaluating the model
    y_pred = predict_labels(w, x_test, threshold=hyperparameters.get("threshold", 0.5))
    accuracy = calculate_accuracy(y_test, y_pred)
    f1_score = calculate_f1_score(y_test, y_pred)

    metrics_dict = {
        "accuracy": accuracy,
        "f1_score": f1_score,
    }

    with open(os.path.join(log_path, f"{unique_id}_metrics.json"), "w") as f:
        json.dump(metrics_dict, f)
    if verbose:
        print(f"Metrics: Accuracy = {metrics_dict['accuracy']:.3f}, F1 Score = {metrics_dict['f1_score']:.3f}")

    print("Training and evaluation complete.")
    if verbose:
        print(f"Results saved in {log_path}")

    return (last_weights, last_loss), metrics_dict


def standardize_model_output(model_output):
    """Standardize model output to always return (w, losses, weights)."""
    if len(model_output) == 3:
        return model_output  # (w, losses, weights)
    elif len(model_output) == 2:
        w, loss = model_output
        return w, [loss], [w]  # Convert single loss and weight to lists
    else:
        raise ValueError("Unexpected model output format.")


def hyperparameters_grid_search(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    model_func: Callable,
    metrics="f1",
    verbose=False,
    **hyperparameter_ranges: Dict[str, list],
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Perform hyperparameter tuning with grid search for the given model function and create performance plots.

    Args:
        x_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        x_test (np.ndarray): Test features.
        y_test (np.ndarray): Test labels.
        model_func (Callable): The model function to train.
        metrics (str): Metric to optimize, defaults to "f1".
        verbose (bool): Whether to print detailed information.
        **hyperparameter_ranges (dict): Ranges of hyperparameters to tune, including threshold.

    Returns:
        np.ndarray: The weights of the best model.
        Dict[str, float]: The best hyperparameters.
    """
    # Ensure valid metrics
    if metrics != "f1":
        raise ValueError("Only 'f1' metric is supported")

    best_f1 = 0
    best_hyperparameters = None
    best_weights = None

    # Initialize result storage
    results = {param: [] for param in hyperparameter_ranges.keys()}
    results["f1_score"] = []

    # Get the model's hyperparameters
    model_signature = inspect.signature(model_func)
    model_params = model_signature.parameters
    model_name = model_func.__name__

    print(f"Model: {model_name}")
    print("=" * 20)
    print()

    # Separate fixed and tunable hyperparameters
    fixed_params = {k: v for k, v in hyperparameter_ranges.items() if not isinstance(v, list) and k in model_params}
    tunable_params = {k: v for k, v in hyperparameter_ranges.items() if isinstance(v, list) and k in model_params}
    threshold_values = hyperparameter_ranges.get("threshold", 0.5)
    if isinstance(threshold_values, (int, float)):
        threshold_values = [threshold_values]  # Convert to list if it's a single value, else it causes issues

    # Generate parameter combinations (excluding threshold)
    param_combinations = list(itertools.product(*tunable_params.values()))

    # Prepare saving path
    unique_id = f"{model_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    base_path = "../results"
    res_path = os.path.join(base_path, "grid_search", unique_id)
    os.makedirs(res_path, exist_ok=True)

    print(f"Results will be saved in {res_path}")

    # Iterate over each combination of tunable parameters
    for param_values in param_combinations:
        current_params = dict(zip(tunable_params.keys(), param_values))
        current_params.update(fixed_params)

        if verbose:
            filtered_params = {k: v for k, v in current_params.items() if k != "initial_w"}
            print("\nTraining with parameters:", filtered_params)

        # Train the model with the current set of hyperparameters
        try:
            w, losses, weights = model_func(y_train, x_train, **current_params)
        except Exception as e:
            error_message = f"Error with parameters {current_params}: {e}"
            print(error_message)
            with open(os.path.join(res_path, "error_log.txt"), "a") as f:
                f.write(error_message + "\n")
            continue  # Skip to the next combination if there's an error

        # Evaluate the model for each threshold without retraining
        for threshold in threshold_values:
            try:
                y_pred = predict_labels(w, x_test, threshold=threshold)
                f1_score = calculate_f1_score(y_test, y_pred)
            except Exception as e:
                error_message = f"Error in prediction with threshold {threshold} for {current_params}: {e}"
                print(error_message)
                with open(os.path.join(res_path, "error_log.txt"), "a") as f:
                    f.write(error_message + "\n")
                continue

            if verbose:
                print(f"F1 Score: {f1_score:.3f} for parameters {filtered_params} with threshold={threshold}")

            # Store the result if successful
            for param, value in current_params.items():
                results[param].append(value)
            results["f1_score"].append(f1_score)
            results.setdefault("threshold", []).append(threshold)

            # Update best model if the current one is better
            if f1_score > best_f1:
                best_f1 = f1_score
                best_hyperparameters = {**current_params, "threshold": threshold}
                best_weights = w
                best_filtered_params = {k: v for k, v in best_hyperparameters.items() if k != "initial_w"}
                print("*" * 20)
                print(f"New best F1 Score: {best_f1:.3f} with parameters {best_filtered_params}")
                print("*" * 20)

        # Convert results dictionary to JSON-compatible format
        results_serializable = {
            key: np.array(value).tolist() if isinstance(value[0], np.ndarray) else value
            for key, value in results.items()
        }

        # Save the JSON-compatible results
        with open(os.path.join(res_path, "results.json"), "w") as f:
            json.dump(results_serializable, f, indent=4)

        print("Intermediate results saved.")

    # Add plot of F1 scores for each parameter r

    if verbose:
        print("Best hyperparameters found:")
        best_filtered_params = {k: v for k, v in best_hyperparameters.items() if k != "initial_w"}
        print(best_filtered_params)
        print(f"Best F1 Score: {best_f1:.3f}")

    return best_weights, best_hyperparameters


def stratified_cross_validation(
    x: np.ndarray,
    y: np.ndarray,
    model_func: Callable,
    n_splits: int = 5,
    verbose: bool = False,
    **hyperparameters: Dict[str, float],
) -> Tuple[float, float, List[float], List[float]]:
    """
    Perform stratified k-fold cross-validation on the given model using F1 score and accuracy.

    Args:
        x (np.ndarray): Features for the entire dataset.
        y (np.ndarray): Labels for the entire dataset.
        model_func (Callable): The model function to train and test.
        n_splits (int): Number of folds for cross-validation.
        verbose (bool): Whether to print detailed information.
        **hyperparameters (dict): Hyperparameters for the model.

    Returns:
        Tuple[float, float, List[float], List[float]]:
            - The average F1 score across folds
            - The average accuracy across folds
            - A list of F1 scores for each fold
            - A list of accuracy scores for each fold
    """
    # Initialize variables to store metrics
    f1_scores = []
    accuracy_scores = []
    unique_classes, y_counts = np.unique(y, return_counts=True)
    class_indices = {cls: np.where(y == cls)[0] for cls in unique_classes}

    # Split indices for each class into n_splits folds
    folds = defaultdict(list)
    for cls in unique_classes:
        indices = class_indices[cls]
        np.random.shuffle(indices)
        splits = np.array_split(indices, n_splits)
        for fold_idx in range(n_splits):
            folds[fold_idx].extend(splits[fold_idx])

    # Perform cross-validation
    for fold_idx in range(n_splits):
        # Split data into training and testing for this fold
        test_indices = np.array(folds[fold_idx])
        train_indices = np.array([i for i in range(len(y)) if i not in test_indices])

        x_train, y_train = x[train_indices], y[train_indices]
        x_test, y_test = x[test_indices], y[test_indices]

        # Add initial_w to hyperparameters
        hyperparameters["initial_w"] = np.zeros(x_train.shape[1])
        hyperparameters["y_val"] = y_test
        hyperparameters["tx_val"] = x_test

        # Train and evaluate model using train_and_test
        (last_weights, last_loss), metrics_dict = train_and_test(
            x_train, y_train, x_test, y_test, model_func, metrics="f1", verbose=verbose, **hyperparameters
        )

        # Extract F1 score and accuracy from the metrics dictionary
        f1_score = metrics_dict.get("f1_score")
        accuracy = metrics_dict.get("accuracy")

        f1_scores.append(f1_score)
        accuracy_scores.append(accuracy)

        if verbose:
            print(f"Fold {fold_idx + 1}/{n_splits} - F1 Score: {f1_score:.4f}, Accuracy: {accuracy:.4f}")

    # Compute average metrics across folds
    average_f1_score = np.mean(f1_scores)
    average_accuracy = np.mean(accuracy_scores)

    return average_f1_score, average_accuracy, f1_scores, accuracy_scores


def grid_search_with_cross_validation(
    x: np.ndarray,
    y: np.ndarray,
    model_func: Callable,
    param_grid: Dict[str, List],
    n_splits: int = 5,
    verbose: bool = False,
) -> Tuple[Dict[str, float], float, float, np.ndarray]:
    """
    Perform grid search with stratified cross-validation to find the best hyperparameters
    based on the best average F1 score across folds and generate visualizations of the tuning.

    Args:
        x (np.ndarray): Features for the entire dataset.
        y (np.ndarray): Labels for the entire dataset.
        model_func (Callable): The model function to train and test.
        param_grid (dict): Dictionary where keys are hyperparameter names, and values are lists of values to try.
        n_splits (int): Number of folds for cross-validation.
        verbose (bool): Whether to print detailed information.

    Returns:
        Tuple containing:
            - Best hyperparameters as a dictionary
            - Best average F1 score across folds
            - Best average accuracy across folds
            - Best model weights
    """
    threshold_values = param_grid.get("threshold", [0.5])
    model_param_grid = {k: v for k, v in param_grid.items() if k != "threshold"}

    best_avg_f1_score = -np.inf
    best_params = None
    best_avg_accuracy = None
    best_weights = None

    tuning_results = {
        "params": [],
        "avg_f1_scores": [],
        "thresholds": [],
        "f1_scores_by_param": defaultdict(lambda: defaultdict(list)),
    }

    model_param_combinations = list(product(*model_param_grid.values()))
    param_names = list(model_param_grid.keys())

    for param_values in model_param_combinations:
        params = dict(zip(param_names, param_values))
        f1_scores_by_threshold = {threshold: [] for threshold in threshold_values}

        unique_classes, y_counts = np.unique(y, return_counts=True)
        class_indices = {cls: np.where(y == cls)[0] for cls in unique_classes}
        folds = defaultdict(list)

        for cls in unique_classes:
            indices = class_indices[cls]
            np.random.shuffle(indices)
            splits = np.array_split(indices, n_splits)
            for fold_idx in range(n_splits):
                folds[fold_idx].extend(splits[fold_idx])

        for fold_idx in range(n_splits):
            test_indices = np.array(folds[fold_idx])
            train_indices = np.array([i for i in range(len(y)) if i not in test_indices])

            x_train, y_train = x[train_indices], y[train_indices]
            x_test, y_test = x[test_indices], y[test_indices]

            params["initial_w"] = np.zeros(x_train.shape[1])
            params["y_val"] = y_test
            params["tx_val"] = x_test

            w, losses, weights = model_func(y_train, x_train, **params)

            for threshold in threshold_values:
                y_pred = predict_labels(w, x_test, threshold=threshold)
                f1_score = calculate_f1_score(y_test, y_pred)
                f1_scores_by_threshold[threshold].append(f1_score)

                if verbose:
                    print(f"Fold {fold_idx + 1}/{n_splits}, Threshold: {threshold}, " f"F1 Score: {f1_score:.4f}")

        for threshold in threshold_values:
            avg_f1_score = np.mean(f1_scores_by_threshold[threshold])

            tuning_results["params"].append(params)
            tuning_results["avg_f1_scores"].append(avg_f1_score)
            tuning_results["thresholds"].append(threshold)

            # Store F1 scores by each parameter individually
            for param_name, param_value in params.items():
                tuning_results["f1_scores_by_param"][threshold][param_name].append((param_value, avg_f1_score))

            if avg_f1_score > best_avg_f1_score:
                best_avg_f1_score = avg_f1_score
                best_params = {**params, "threshold": threshold}
                best_weights = w

                if verbose:
                    filtered_params = {
                        k: v for k, v in best_params.items() if k not in ["initial_w", "y_val", "tx_val"]
                    }
                    print("*" * 20)
                    print(f"New best F1 Score: {best_avg_f1_score:.4f} with parameters {filtered_params}")
                    print("*" * 20)

    # Results directory
    unique_id = f"{model_func.__name__}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    base_path = "../results"
    res_path = os.path.join(base_path, "grid_search_cv", unique_id)
    os.makedirs(res_path, exist_ok=True)

    # Plotting F1-Score against each parameter for each threshold
    for param_name in model_param_grid.keys():
        plt.figure(figsize=(10, 6))

        for threshold in threshold_values:
            if param_name in tuning_results["f1_scores_by_param"][threshold]:
                values_and_scores = tuning_results["f1_scores_by_param"][threshold][param_name]
                values, scores = zip(*values_and_scores)
                plt.plot(values, scores, marker="o", label=f"Threshold = {threshold}")

        plt.xlabel(param_name)
        plt.ylabel("F1 Score")
        plt.title(f"F1 Score vs {param_name} for different thresholds")
        plt.legend(title="Threshold")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(res_path, f"{param_name}_f1_scores.png"))

    if verbose:
        # Filter out 'initial_w', 'y_val', and 'tx_val' from the parameters to be printed
        best_params_filtered = {k: v for k, v in best_params.items() if k not in ["initial_w", "y_val", "tx_val"]}
        print("\nBest Parameters:", best_params_filtered)
        print(f"Best Average F1 Score: {best_avg_f1_score:.4f}")

    return best_params, best_avg_f1_score, best_avg_accuracy, best_weights


def train(x_train, y_train, model_func, verbose=False, **hyperparameters):
    """
    Train a model with the given hyperparameters and return the weights and losses, using the whole training set.

    Args:
        x_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        model_func (callable): The model function to train.
        verbose (bool): Whether to print detailed information.
        **hyperparameters: Hyperparameters for the model.

    Returns:
        np.ndarray: The weights of the trained model.
        List[float]: The training losses at each iteration.
    """
    # Extract hyperparameters for the model and check for errors
    model_signature = inspect.signature(model_func)
    model_params = model_signature.parameters

    model_specific_params = {k: v for k, v in hyperparameters.items() if k in model_params}
    if verbose:
        print("Model specific parameters:")
        for k, v in model_specific_params.items():
            print(f"    {k}: {v}")
        print()

    # Train the model
    start_time = time.time()
    print("Training model...")
    model_output = model_func(y_train, x_train, **model_specific_params)
    w, losses, _ = standardize_model_output(model_output)
    end_time = time.time()

    training_time = end_time - start_time
    print(f"Training done in {training_time:.2f} s with final loss: {losses[-1]:.5f}")

    # Create training loss plot
    # Display if verbose is True
    # Save the plot either way in the results folder

    if len(losses) > 1:
        plt.plot(losses, color="blue")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.legend(["Training Loss"])

        if verbose:
            plt.show()

        # Save the plot
        base_path = "../results"
        log_path = os.path.join(base_path, "logs", "train")
        os.makedirs(log_path, exist_ok=True)

        plt.savefig(os.path.join(log_path, "training_loss_plot.png"))
        plt.close()

    return w, losses
