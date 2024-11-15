# ADD SMALL DESCRIPTION HERE
# -----------------------------------------------------------------------------

import csv
import json
import os
import sys

import numpy as np


def preprocess_features(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    feature_names: np.ndarray,
    feature_dict: dict,
    percentage=0.9,
    imputation_strategy="mean",
    data_path="../data",
    save_data=False,
):
    """
    Preprocess non-structured training and test data using the information from feature_dict.

    Args:
        x_train (np.array): Training data (samples as rows, features as columns).
        y_train (np.array): Training labels.
        x_test (np.array): Test data (samples as rows, features as columns).
        feature_names (np.array): Array of feature names, index corresponds to columns in x_train and x_test.
        feature_dict (dict): Dictionary describing how to process each feature.
        percentage (float): Threshold for the percentage of missing values allowed before dropping a feature.
        imputation_strategy (str): Imputation strategy for missing values in numerical features, either 'mean' or 'median'.

    Returns:
        x_train_preprocessed (np.array): Preprocessed training data.
        x_test_preprocessed (np.array): Preprocessed test data.
        feature_names_processed (list): List of feature names after preprocessing.
        processed_metadata (dict): Dictionary containing metadata for each feature after preprocessing.
    """
    print("Preprocessing data...")
    x_train_processed = []
    x_test_processed = []
    feature_names_processed = []
    processed_metadata = {}

    for i, feature_name in enumerate(feature_names):
        # Get the metadata for the feature
        feature_info = feature_dict.get(feature_name, {})

        # Check if marked for discard
        if feature_info.get("discard", False):
            continue

        train_feature_col = x_train[:, i]
        test_feature_col = x_test[:, i]

        # Replace invalid values with NaN
        invalid_values = feature_info.get("invalid_values", [])
        train_feature_col = np.where(np.isin(train_feature_col, invalid_values), np.nan, train_feature_col)
        test_feature_col = np.where(np.isin(test_feature_col, invalid_values), np.nan, test_feature_col)

        # Check percentage of missing values in the training data
        missing_values_train = np.sum(np.isnan(train_feature_col)) / len(train_feature_col)
        if missing_values_train > percentage:
            continue

        # Handle special cases # NOT SURE IT'S WORKING
        special_cases = feature_info.get("special_cases", {})
        if special_cases:
            for invalid_val, replacement in special_cases.items():
                train_feature_col = np.where(train_feature_col == invalid_val, replacement, train_feature_col)
                test_feature_col = np.where(test_feature_col == invalid_val, replacement, test_feature_col)

        feature_type = feature_info.get("type")
        encoding = feature_info.get("encoding")

        if feature_type == "numerical" and encoding == "none":
            if imputation_strategy == "mean":
                impute_value = np.nanmean(train_feature_col)
            elif imputation_strategy == "median":
                impute_value = np.nanmedian(train_feature_col)
            else:
                raise ValueError("Imputation strategy must be either 'mean' or 'median'.")

            train_feature_col[np.isnan(train_feature_col)] = impute_value
            test_feature_col[np.isnan(test_feature_col)] = impute_value

            train_feature_col, test_feature_col = standardize_feature(train_feature_col, test_feature_col)
            x_train_processed.append(train_feature_col)
            x_test_processed.append(test_feature_col)
            feature_names_processed.append(feature_name)
            processed_metadata[feature_name] = {"type": "numerical", "encoding": "standardized"}
            continue

        if encoding == "one-hot":  # Not imputing missing values for one-hot encoded features
            train_one_hot, test_one_hot, feature_one_hot = one_hot_encode_feature(
                train_feature_col, test_feature_col, feature_name
            )
            x_train_processed.append(train_one_hot)
            x_test_processed.append(test_one_hot)
            for feature in feature_one_hot:
                feature_names_processed.append(feature)
                processed_metadata[feature] = {"type": "categorical", "encoding": "one-hot"}
            continue

        elif feature_type == "categorical":
            # Find the most frequent value in the training data
            unique_vals, counts = np.unique(train_feature_col[~np.isnan(train_feature_col)], return_counts=True)
            mode_value = unique_vals[np.argmax(counts)]

            train_feature_col[np.isnan(train_feature_col)] = mode_value
            test_feature_col[np.isnan(test_feature_col)] = mode_value

            if encoding == "none":
                x_train_processed.append(train_feature_col)
                x_test_processed.append(test_feature_col)
                feature_names_processed.append(feature_name)
                processed_metadata[feature_name] = {"type": "categorical", "encoding": "ordinal"}
                continue

            elif encoding == "binary":
                train_binary, test_binary = binary_encode_feature(train_feature_col, test_feature_col, feature_name)
                x_train_processed.append(train_binary)
                x_test_processed.append(test_binary)
                feature_names_processed.append(feature_name)
                processed_metadata[feature_name] = {"type": "categorical", "encoding": "binary"}
                continue

            elif encoding == "frequency":
                train_freq, test_freq = frequency_encode_feature(train_feature_col, test_feature_col)
                x_train_processed.append(train_freq)
                x_test_processed.append(test_freq)
                feature_names_processed.append(feature_name)
                processed_metadata[feature_name] = {"type": "categorical", "encoding": "frequency"}
                continue

            elif encoding == "past_30_days":
                train_freq, test_freq = past_30_days_encode_feature(train_feature_col, test_feature_col)
                x_train_processed.append(train_freq)
                x_test_processed.append(test_freq)
                feature_names_processed.append(feature_name)
                processed_metadata[feature_name] = {"type": "categorical", "encoding": "past_30_days"}
                continue

            elif encoding == "health_check":
                train_freq, test_freq = health_check_encode_feature(train_feature_col, test_feature_col)
                x_train_processed.append(train_freq)
                x_test_processed.append(test_freq)
                feature_names_processed.append(feature_name)
                processed_metadata[feature_name] = {"type": "categorical", "encoding": "health_check"}
                continue

    x_train_processed = [col.reshape(-1, 1) if col.ndim == 1 else col for col in x_train_processed]
    x_test_processed = [col.reshape(-1, 1) if col.ndim == 1 else col for col in x_test_processed]

    # Concatenate all processed data back together
    x_train_preprocessed = np.hstack(x_train_processed)
    x_test_preprocessed = np.hstack(x_test_processed)

    # Save the preprocessed data
    if save_data:
        folder_name = "preprocessed"
        if imputation_strategy == "mean" and percentage == 0.9:
            folder_name = os.path.join(folder_name, "std")
        else:
            folder_name = os.path.join(folder_name, f"{imputation_strategy}_{percentage}")

        save_path = os.path.join(data_path, folder_name)
        os.makedirs(save_path, exist_ok=True)

        np.savetxt(os.path.join(save_path, "x_train.csv"), x_train_preprocessed, delimiter=",")
        np.savetxt(os.path.join(save_path, "y_train.csv"), y_train, delimiter=",", fmt="%d")
        np.savetxt(os.path.join(save_path, "x_test.csv"), x_test_preprocessed, delimiter=",")
        np.savetxt(os.path.join(save_path, "feature_names.csv"), feature_names_processed, delimiter=",", fmt="%s")
        with open(os.path.join(save_path, "feature_dict.json"), "w") as file:
            json.dump(processed_metadata, file)

        print(f"Data preprocessed and saved in {save_path}.")
    else:
        print("Data preprocessed.")

    return x_train_preprocessed, x_test_preprocessed, feature_names_processed, processed_metadata


def standardize_feature(
    train_feature_col: np.ndarray, test_feature_col: np.ndarray
):  # Maybe add other standardization methods for numerical features ?
    """
    Standardize a numerical feature w

    Args:
        train_feature_col (np.array): Training feature column.
        test_feature_col (np.array): Test feature column.

    Returns:
        train_feature_col (np.array): Standardized training feature column.
        test_feature_col (np.array): Standardized test feature column.
    """
    mean = np.nanmean(train_feature_col)
    std = np.nanstd(train_feature_col)

    train_feature_col = (train_feature_col - mean) / std
    test_feature_col = (test_feature_col - mean) / std

    return train_feature_col, test_feature_col


def one_hot_encode_feature(train_feature_col: np.ndarray, test_feature_col: np.ndarray, feature_name: str):
    """
    One-hot encode a categorical feature.

    Args:
        train_feature_col (np.array): Training feature column.
        test_feature_col (np.array): Test feature column.
        feature_name (str): Name of the feature.

    Returns:
        train_one_hot (np.array): One-hot encoded training feature column.
        test_one_hot (np.array): One-hot encoded test feature column.
        feature_one_hot (list): List of feature names for each one-hot encoded column.
    """
    # Find the unique values across both training and test sets, excluding NaNs
    all_data = np.concatenate([train_feature_col, test_feature_col])
    unique_values = np.unique(all_data[~np.isnan(all_data)])

    # Create a dictionary that maps each unique value to a one-hot index
    value_to_index = {value: idx for idx, value in enumerate(unique_values)}

    # One-hot encoding
    train_one_hot = np.zeros((train_feature_col.shape[0], len(unique_values)))
    for i, value in enumerate(train_feature_col):
        if not np.isnan(value):
            train_one_hot[i, value_to_index[value]] = 1

    test_one_hot = np.zeros((test_feature_col.shape[0], len(unique_values)))
    for i, value in enumerate(test_feature_col):
        if not np.isnan(value):
            test_one_hot[i, value_to_index[value]] = 1

    # Feature names for each one-hot encoded column
    feature_one_hot = [f"{feature_name}_{val}" for val in unique_values]

    return train_one_hot, test_one_hot, feature_one_hot


def binary_encode_feature(train_feature_col: np.ndarray, test_feature_col: np.ndarray, feature_name: str):
    """
    Binary encode a categorical feature as 0 or 1.

    Args:
        train_feature_col (np.array): Training feature column.
        test_feature_col (np.array): Test feature column.
        feature_name (str): Name of the feature.

    Returns:
        train_binary (np.array): Binary encoded training feature column.
        test_binary (np.array): Binary encoded test feature column.
    """
    # Combine the training and test data to ensure all categories are present
    all_data = np.concatenate([train_feature_col, test_feature_col])
    unique_values = np.unique(all_data[~np.isnan(all_data)])

    # Assert that there are only two unique values
    assert len(unique_values) == 2, f"Error: {feature_name} has more than two unique values."

    # Map the two unique values to 0 and 1
    value_to_binary = {unique_values[0]: 0, unique_values[1]: 1}

    # Binary encoding
    train_binary = np.array([value_to_binary[value] if not np.isnan(value) else np.nan for value in train_feature_col])
    test_binary = np.array([value_to_binary[value] if not np.isnan(value) else np.nan for value in test_feature_col])

    return train_binary, test_binary


def map_value(value, encoding_map):
    """Map a single value based on the encoding map."""
    if np.isnan(value):
        return np.nan

    # Iterate over the encoding_map and apply the correct encoding
    for condition, encoded_value in encoding_map.items():
        if isinstance(condition, tuple):
            if condition[0] <= value <= condition[1]:
                return encoded_value
        elif isinstance(condition, (int, float)) and value == condition:
            return encoded_value
    return np.nan


def frequency_encode_feature(train_feature_col: np.ndarray, test_feature_col: np.ndarray):
    """
    Frequency encode a categorical feature by replacing each category with its frequency in the training data.

    Args:
        train_feature_col (np.array): Training feature column.
        test_feature_col (np.array): Test feature column.

    Returns:
        train_freq (np.array): Frequency encoded training feature column.
        test_freq (np.array): Frequency encoded test feature column.
    """
    encoding_map = {
        (101, 199): 1,
        (201, 299): 2,
        300: 3,
        (301, 399): 4,
        0: 5,
        555: 5,
    }

    train_feature = np.array([map_value(value, encoding_map) for value in train_feature_col])
    test_feature = np.array([map_value(value, encoding_map) for value in test_feature_col])

    return train_feature, test_feature


def past_30_days_encode_feature(train_feature_col: np.ndarray, test_feature_col: np.ndarray):
    """
    Past 30 days encode a categorical feature by replacing each category with its frequency in the training data.

    Args:
        train_feature_col (np.array): Training feature column.
        test_feature_col (np.array): Test feature column.

    Returns:
        train_freq (np.array): Frequency encoded training feature column.
        test_freq (np.array): Frequency encoded test feature column.
    """
    encoding_map = {
        (101, 199): 1,
        (201, 299): 2,
        888: 3,
        0: 3,
    }

    train_feature = np.array([map_value(value, encoding_map) for value in train_feature_col])
    test_feature = np.array([map_value(value, encoding_map) for value in test_feature_col])

    return train_feature, test_feature


def health_check_encode_feature(train_feature_col: np.ndarray, test_feature_col: np.ndarray):
    encoding_map = {
        (101, 199): 1,
        (201, 299): 2,
        (301, 399): 3,
        (401, 499): 4,
        888: 5,
        0: 5,
        555: 5,
    }

    train_feature = np.array([map_value(value, encoding_map) for value in train_feature_col])
    test_feature = np.array([map_value(value, encoding_map) for value in test_feature_col])

    return train_feature, test_feature
