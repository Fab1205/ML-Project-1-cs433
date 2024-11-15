# Imports
import os
import sys
from datetime import datetime

import numpy as np

from data_loading import load_data
from helpers import create_csv_submission_custom
from models import logistic_regression
from train_pipeline import train
from utils import predict_labels

# Load and preprocessed data
data_path = "../data"
x_train, x_test, y_train, train_ids, test_ids, feature_names, feature_dict = load_data(data_path, preprocessed=True)

# Hyperparameters
hyperparameters_best_model = {}
hyperparameters_best_model["gamma"] = 0.01
hyperparameters_best_model["threshold"] = 0.6
hyperparameters_best_model["class_weight"] = 7
hyperparameters_best_model["max_iters"] = 50000  # Might take a while
hyperparameters_best_model["patience"] = 50
hyperparameters_best_model["initial_w"] = np.zeros(x_train.shape[1])

# Training on the whole dataset
w, loss = train(x_train, y_train, logistic_regression, verbose=True, **hyperparameters_best_model)

# Pipeline for creating subsmission
y_pred = predict_labels(w, x_test, threshold=hyperparameters_best_model["threshold"])

# Labels back to -1 and 1
y_pred = np.where(y_pred == 0, -1, 1)

# unique id = 'logistic_regression' + current timestamp, ie, day monthe hour minute
unique_id = f"logistic_regression_{datetime.now().strftime('%Y%m%d%H%M')}"
print(f"Unique ID: {unique_id}")
create_csv_submission_custom(test_ids, y_pred, unique_id)
