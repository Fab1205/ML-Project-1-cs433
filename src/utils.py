# ADD SMALL DESCRIPTION HERE
# -----------------------------------------------------------------------------


import matplotlib.pyplot as plt
import numpy as np

from implementations import sigmoid


def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Calculate the accuracy of the model.

    Args:
        y_true (np.array): true labels
        y_pred (np.array): predicted labels

    Returns:
        float: accuracy
    """
    return np.mean(y_true == y_pred)


def calculate_f1_score(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Calculate the F1 score of the model.

    Args:
        y_true (np.array): true labels
        y_pred (np.array): predicted labels

    Returns:
        float: F1 score
    """
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return f1


def predict_labels(w: np.ndarray, x_val: np.ndarray, threshold=0.5):
    """
    Generate class predictions using the learned logistic regression model.

    Args:
        w (np.array): weights
        x_val (np.array): features
        threshold (float): threshold for classification

    Returns:
        np.array: predicted labels
    """

    y_val_pred = sigmoid(x_val @ w)

    y_val_pred_labels = np.where(y_val_pred >= threshold, 1, 0)
    return y_val_pred_labels


def stratified_split(
    x_train: np.ndarray, y_train: np.ndarray, train_ids: np.ndarray, validation_ratio=0.2, random_state=42
):
    """
    Split the training data into training and validation sets while maintaining class balance.

    Args:
        x_train (np.ndarray): Training data (samples as rows, features as columns).
        y_train (np.ndarray): Binary target values (0 and 1).
        train_ids (np.ndarray): Array of IDs corresponding to each sample in x_train.
        validation_ratio (float): Ratio of samples to use for validation.
        random_state (int): Random seed for reproducibility.

    Returns:
        x_train_new (np.ndarray): Training data after removing validation samples.
        y_train_new (np.ndarray): Binary target values after removing validation samples.
        train_ids_new (np.ndarray): Array of IDs after removing validation samples.
        x_val (np.ndarray): Validation data.
        y_val (np.ndarray): Binary target values for validation data.
        val_ids (np.ndarray): Array of IDs for validation data.
    """
    np.random.seed(random_state)

    # Separate data by class
    labels = np.unique(y_train)
    class_0_indices = np.where(y_train == labels[0])[0]
    class_1_indices = np.where(y_train == labels[1])[0]

    # Shuffle indices
    np.random.shuffle(class_0_indices)
    np.random.shuffle(class_1_indices)

    # Determine the number of validation samples for each class
    n_val_class_1 = int(len(class_0_indices) * validation_ratio)
    n_val_class_neg1 = int(len(class_1_indices) * validation_ratio)

    # Split into training and validation indices
    val_indices = np.concatenate([class_0_indices[:n_val_class_1], class_1_indices[:n_val_class_neg1]])
    train_indices = np.concatenate([class_0_indices[n_val_class_1:], class_1_indices[n_val_class_neg1:]])

    # Create new training and validation sets
    x_train_new = x_train[train_indices]
    y_train_new = y_train[train_indices]
    train_ids_new = train_ids[train_indices]

    x_val = x_train[val_indices]
    y_val = y_train[val_indices]
    val_ids = train_ids[val_indices]

    # Return new training and validation sets along with their respective IDs
    return x_train_new, y_train_new, train_ids_new, x_val, y_val, val_ids


def entropy(values):
    """Calculate entropy of a binary or categorical feature."""
    _, counts = np.unique(values, return_counts=True)
    probs = counts / len(values)
    return -np.sum(probs * np.log2(probs + 1e-9))


def conditional_entropy(feature, target):
    """Calculate conditional entropy H(X|Y) of a feature given a binary target."""
    unique_target_values = np.unique(target)
    cond_entropy = 0
    for value in unique_target_values:
        subset = feature[target == value]
        cond_entropy += (len(subset) / len(feature)) * entropy(subset)
    return cond_entropy


def mutual_information(x, y, display_plot=False):
    """
    Compute mutual information between each feature in x and binary target y.

    Args:
        x (np.ndarray): Feature matrix (samples as rows, features as columns).
        y (np.ndarray): Binary target variable (1D array with values 0 or 1).
        display_plot (bool): If True, displays a bar plot of mutual information values.

    Returns:
        mi_values (list): List of mutual information values for each feature.
    """
    mi_values = []
    for i in range(x.shape[1]):
        h_x = entropy(x[:, i])
        h_x_given_y = conditional_entropy(x[:, i], y)
        mi = h_x - h_x_given_y
        mi_values.append(mi)

    # Plot mutual information values if requested
    if display_plot:
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(mi_values)), mi_values)
        plt.xlabel("Features")
        plt.ylabel("Mutual Information with Target")
        plt.title("Mutual Information between Features and Target")
        plt.show()

    return mi_values


def feature_target_correlation(x_train: np.ndarray, y_train: np.ndarray, feature_names: np.ndarray, display_plot=False):
    """
    Calculate the correlation between each feature and the target variable (binary classification)
    and optionally display a bar chart of feature-target correlations.

    Args:
        x_train (np.ndarray): Training data (samples as rows, features as columns).
        y_train (np.ndarray): Binary target values (0 and 1).
        feature_names (np.ndarray): Array of feature names, matching the columns of `x_train`.
        display_plot (bool): If True, display a sorted bar chart of feature-target correlations.

    Returns:
        feature_target_corr (dict): Dictionary with feature names as keys and correlations as values.
    """
    feature_target_corr = {}

    # Calculate correlation for each feature with the target
    for i, feature_name in enumerate(feature_names):
        correlation = np.corrcoef(x_train[:, i], y_train)[0, 1]
        feature_target_corr[feature_name] = correlation

    # Display bar chart of feature-target correlations if display_plot is True
    if display_plot:
        sorted_corr = dict(sorted(feature_target_corr.items(), key=lambda item: abs(item[1]), reverse=True))
        plt.figure(figsize=(12, 6))
        plt.bar(sorted_corr.keys(), sorted_corr.values())
        plt.xlabel("Features")
        plt.ylabel("Correlation with Target")
        plt.title("Feature-Target Correlation (Sorted by Magnitude)")
        plt.xticks(rotation=90)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()

    return feature_target_corr


def point_biserial_correlation(x, y, feature_dict, display_plot=False):
    """
    Compute the point-biserial correlation between each numerical feature in x and binary target y.

    Args:
        x (np.ndarray): Feature matrix (samples as rows, features as columns).
        y (np.ndarray): Binary target variable (1D array with values 0 or 1).
        feature_dict (dict): Dictionary with metadata about each feature, keyed by feature name.
        display_plot (bool): If True, displays a bar plot of point-biserial correlation values.

    Returns:
        pb_values (dict): Dictionary of point-biserial correlations for numerical features.
    """
    pb_values = {}

    for i, (feature_name, metadata) in enumerate(feature_dict.items()):
        if metadata["type"] == "numerical":
            y_0 = x[y == 0, i]
            y_1 = x[y == 1, i]
            mean_0 = np.mean(y_0)
            mean_1 = np.mean(y_1)
            numerator = (mean_1 - mean_0) * np.sqrt(len(y_0) * len(y_1))
            denominator = np.std(x[:, i]) * np.sqrt(len(y))
            point_biserial_corr = numerator / denominator
            pb_values[feature_name] = point_biserial_corr

    # Plot point-biserial values if requested
    if display_plot:
        plt.figure(figsize=(10, 6))
        plt.bar(pb_values.keys(), pb_values.values())
        plt.xlabel("Features")
        plt.ylabel("Point-Biserial Correlation with Target")
        plt.title("Point-Biserial Correlation between Numerical Features and Target")
        plt.xticks(rotation=90)
        plt.show()

    return pb_values


def correlation_matrix(x_train: np.ndarray, feature_names: np.ndarray, display_plot=False):
    """
    Calculate the correlation matrix between features and optionally display it.

    Args:
        x_train (np.ndarray): Training data (samples as rows, features as columns).
        feature_names (np.ndarray): Array of feature names, matching the columns of `x_train`.
        display_plot (bool): If True, display a heatmap of the correlation matrix.

    Returns:
        corr_matrix (np.ndarray): Correlation matrix.
    """
    # Compute the correlation matrix
    corr_matrix = np.corrcoef(x_train, rowvar=False)

    # Plot the correlation matrix if requested
    if display_plot:
        plt.figure(figsize=(10, 8))
        plt.imshow(corr_matrix, cmap="coolwarm", interpolation="nearest")
        plt.colorbar(label="Correlation Coefficient")
        plt.xticks(range(len(feature_names)), feature_names, rotation=90)
        plt.yticks(range(len(feature_names)), feature_names)
        plt.title("Feature-Feature Correlation Matrix")
        plt.tight_layout()
        plt.show()

    return corr_matrix
