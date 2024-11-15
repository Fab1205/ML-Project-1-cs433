import json
import os

import numpy as np

from data_processing import preprocess_features


def load_data(data_path: str, preprocessed=False):

    def load_csv(filename, skip_header=True, dtype=float):
        """Utility function to load a CSV file."""
        return np.genfromtxt(os.path.join(data_path, filename), delimiter=",", skip_header=skip_header, dtype=dtype)

    print("Loading raw data...")
    feature_dict = load_features_dict(os.path.join(data_path, "feature_dict.json"))
    data_path = os.path.join(data_path, "raw/dataset")
    feature_names = load_csv("x_train.csv", skip_header=False, dtype=str)[0, 1:]
    y_train = load_csv("y_train.csv", dtype=int, skip_header=True)[:, 1]
    x_train = load_csv("x_train.csv", skip_header=True)
    x_test = load_csv("x_test.csv", skip_header=True)
    print("Data loaded.")

    # Change labels to 0 and 1
    y_train[y_train == -1] = 0

    # Extract IDs and features
    train_ids, x_train = x_train[:, 0].astype(int), x_train[:, 1:]
    test_ids, x_test = x_test[:, 0].astype(int), x_test[:, 1:]

    if preprocessed:
        x_train, x_test, feature_names, feature_dict = preprocess_features(
            x_train, y_train, x_test, feature_names, feature_dict
        )

    return x_train, x_test, y_train, train_ids, test_ids, feature_names, feature_dict


def load_features_dict(json_path: str):
    """
    Load feature dictionary from a JSON file.

    Args:
        json_path (str): Path to the JSON file.

    Returns:
        features_dict (dict): Dictionary containing feature metadata (type and encoding).
    """
    with open(json_path, "r") as file:
        return json.load(file)
