import numpy as np

from utils import correlation_matrix, mutual_information, point_biserial_correlation


def analyze_features_dict(feature_dict_: dict):
    """
    Analyze the feature dictionary and print summary information.

    Args:
        feature_dict (dict): Dictionary containing feature metadata.
    """
    # Initialize counters
    total_features = len(feature_dict_)
    categorical_count = 0
    numerical_count = 0
    discarded_count = 0

    categorical_subtypes = {}
    numerical_subtypes = {}
    encoding_count = {}

    # Iterate through each feature in the dictionary
    for feature, props in feature_dict_.items():
        # Count discarded features
        if props.get("discard", False):
            discarded_count += 1

        # Count categorical vs numerical features
        if props.get("type") == "categorical":
            categorical_count += 1
            # Count subtypes for categorical features
            subtype = props.get("subtype", "undefined")
            categorical_subtypes[subtype] = categorical_subtypes.get(subtype, 0) + 1

            # Count encoding types for categorical features
            encoding = props.get("encoding", "none")
            encoding_count[encoding] = encoding_count.get(encoding, 0) + 1

        elif props.get("type") == "numerical":
            numerical_count += 1
            # Count subtypes for numerical features
            subtype = props.get("subtype", "undefined")
            numerical_subtypes[subtype] = numerical_subtypes.get(subtype, 0) + 1

        # Print the results
    print("Features Dict Information")
    print("=" * 20)
    print(f"\nTotal number of features: {total_features}")
    print(f"\nNumber of categorical features: {categorical_count}")
    print(f"Number of numerical features: {numerical_count}")
    print(f"\nNumber of features to be discarded: {discarded_count}")

    print("\nCategorical feature subtypes:")
    for subtype, count in categorical_subtypes.items():
        print(f"  {subtype}: {count}")

    print("\nNumerical feature subtypes:")
    for subtype, count in numerical_subtypes.items():
        print(f"  {subtype}: {count}")

    print("\nEncoding types:")
    for encoding, count in encoding_count.items():
        print(f"  {encoding}: {count}")


def print_dataset_info(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, feature_dict_: dict):
    """
    Prints summary information about the datasets (train/test) and features.

    Args:
        x_train (np.array): Training features.
        y_train (np.array): Training labels.
        x_test (np.array): Test features.
        feature_dict_ (dict): Dictionary containing feature metadata.
    """

    print("Dataset Information")
    print("=" * 20)

    # Training data info
    print("\nTraining Data")
    print(f"{x_train.shape[0]} samples")
    print(f"With labels: {np.unique(y_train)}")
    print(f"{np.unique(y_train)[0]}: {np.sum(y_train == np.unique(y_train)[0]) / len(y_train) * 100:.2f}%")
    print(f"and {np.unique(y_train)[1]}: {np.sum(y_train == np.unique(y_train)[1]) / len(y_train) * 100:.2f}%")

    # Test data info
    print("\nTest Data")
    print(f"{x_test.shape[0]} samples")

    # Feature names info
    print("\nFeatures")
    print(f"{len(feature_dict_)} features:")
    print('The first column of the datasets is the "sample id" and is not in the dictionary\n')


def check_missing_values(x_train: np.ndarray, x_test: np.ndarray, feature_names: list, p=0.9, print_features=False):
    """
    Check for missing values in the datasets and print summary information.

    Args:
        x_train (np.array): Training features.
        x_test (np.array): Test features.
        feature_names (list): List of feature names.
        p (float): Threshold for missing values percentage.
        print_features (bool): If True, print features with missing values exceeding the threshold.
    """
    # Overall missing values count
    total_missing_train = np.isnan(x_train).sum()
    total_missing_test = np.isnan(x_test).sum()
    total_missing_both = total_missing_train + total_missing_test

    print("Overall missing values:")
    print(f" - x_train missing values: {total_missing_train} ({total_missing_train / x_train.size * 100:.2f}%)")
    print(f" - x_test missing values: {total_missing_test} ({total_missing_test / x_test.size * 100:.2f}%)")
    print(f" - Combined missing values: {total_missing_both}\n")

    # Per-feature missing values count, only for features with missing values or exceeding the threshold `p`
    print(f"Features with more than {p* 100:.2f}% missing values:")
    missing_features_count_tot = 0
    missing_features_count_p = 0
    for i, feature in enumerate(feature_names):
        missing_train = np.isnan(x_train[:, i]).sum()
        if missing_train > 0:
            missing_features_count_tot += 1
        if missing_train / x_train.shape[0] > p:
            missing_features_count_p += 1
            if print_features:
                print(f" - {feature}: {missing_train} ({missing_train / x_train.shape[0] * 100:.2f}%)")

    print(f"Total features with more than {p* 100:.2f}% missing values: {missing_features_count_p}")
    print(f"Total features with missing values: {missing_features_count_tot}")


def update_feature_dict(feature_dict, feature_names):
    """
    Update the feature dictionary to retain only the entries in feature_names.

    Args:
        feature_dict (dict): Original dictionary containing metadata for each feature.
        feature_names (list): List of feature names to retain in the updated dictionary.

    Returns:
        updated_feature_dict (dict): Updated dictionary containing only the features in feature_names.
    """
    # Create a new dictionary with only the selected feature names
    updated_feature_dict = {name: feature_dict[name] for name in feature_names if name in feature_dict}

    return updated_feature_dict


def select_features_based_on_correlation(
    x_train,
    y_train,
    x_test,
    feature_names,
    feature_dict,
    pb_threshold=None,
    mi_threshold=None,
    ff_threshold=None,
    display_plot=False,
):
    """
    Select features based on point-biserial, mutual information, and feature-feature correlation thresholds.
    Skips correlation type if the corresponding threshold is set to None.

    Args:
        x_train (np.ndarray): Training data (samples as rows, features as columns).
        x_test (np.ndarray): Test data (samples as rows, features as columns).
        y_train (np.ndarray): Binary target variable for training data (1D array with values 0 or 1).
        feature_names (np.ndarray or list): Array of feature names, matching the columns of `x_train`.
        feature_dict (dict): Dictionary with metadata about each feature, keyed by feature name.
        pb_threshold (float or None): Threshold for point-biserial correlation with the binary target.
                                      If None, skip point-biserial selection.
        mi_threshold (float or None): Threshold for mutual information with the binary target.
                                      If None, skip mutual information selection.
        ff_threshold (float or None): Threshold for feature-feature correlation.
                                      If None, skip feature-feature correlation selection.
        display_plot (bool): If True, display plots for each type of correlation used.

    Returns:
        selected_x_train (np.ndarray): Training set with only selected features.
        selected_x_test (np.ndarray): Test set with only selected features.
        selected_feature_names (np.ndarray): Array of names for the selected features.
        selected_feature_dict (dict): Updated feature dictionary with only the selected features.
    """
    # Ensure feature_names is a NumPy array for proper indexing
    feature_names = np.array(feature_names)

    # Track selected features and discarded features
    selected_features = []
    selected_feature_names = []
    discarded_features = set()

    # 1. Calculate Point-Biserial Correlations if pb_threshold is provided
    if pb_threshold is not None:
        pb_values = point_biserial_correlation(x_train, y_train, feature_dict, display_plot=display_plot)
        for i, feature_name in enumerate(feature_names):
            if feature_name in pb_values and abs(pb_values[feature_name]) >= pb_threshold:
                selected_features.append((x_train[:, i], x_test[:, i]))
                selected_feature_names.append(feature_name)

    # 2. Calculate Mutual Information if mi_threshold is provided
    if mi_threshold is not None:
        mi_values = mutual_information(x_train, y_train, display_plot=display_plot)
        for i, mi_value in enumerate(mi_values):
            if mi_value >= mi_threshold and feature_names[i] not in selected_feature_names:
                selected_features.append((x_train[:, i], x_test[:, i]))
                selected_feature_names.append(feature_names[i])

    # 3. Calculate Feature-Feature Correlation if ff_threshold is provided
    if ff_threshold is not None:
        ff_corr_matrix = correlation_matrix(x_train, feature_names, display_plot=display_plot)
        for i in range(ff_corr_matrix.shape[0]):
            if i in discarded_features:
                continue
            for j in range(i + 1, ff_corr_matrix.shape[1]):
                if j in discarded_features:
                    continue

                feature_name_i = feature_names[i]
                feature_name_j = feature_names[j]

                # Check if correlation exceeds threshold
                if abs(ff_corr_matrix[i, j]) > ff_threshold:

                    # Combine features if both are not yet discarded
                    if feature_name_i not in discarded_features and feature_name_j not in discarded_features:
                        # Combine features for train and test sets by averaging
                        combined_feature_train = (x_train[:, i] + x_train[:, j]) / 2
                        combined_feature_test = (x_test[:, i] + x_test[:, j]) / 2
                        combined_feature_name = f"{feature_name_i}_{feature_name_j}_combined"

                        # Add combined feature to selected features
                        selected_features.append((combined_feature_train, combined_feature_test))
                        selected_feature_names.append(combined_feature_name)

                        # Discard the second feature in the correlated pair
                        discarded_features.add(j)
                else:
                    # If no high correlation, retain the original feature if not already added
                    if feature_name_i not in selected_feature_names:
                        selected_features.append((x_train[:, i], x_test[:, i]))
                        selected_feature_names.append(feature_name_i)

    # Format selected features into np.ndarrays for train and test sets
    selected_x_train = np.array([feat[0] for feat in selected_features]).T
    selected_x_test = np.array([feat[1] for feat in selected_features]).T

    # Update the feature dictionary to retain only the selected features
    selected_feature_dict = update_feature_dict(feature_dict, selected_feature_names)

    return selected_x_train, selected_x_test, np.array(selected_feature_names), selected_feature_dict
