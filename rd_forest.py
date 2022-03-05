"""
Build and train a random forest machine learning algorithm.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from hyperopt import fmin, tpe, hp, STATUS_OK

import mlflow
import mlflow.sklearn

from helper_functions import eval_model

mlflow.set_experiment('DataExploration_Project - Random Forest')

X_train = pd.read_csv('data/output/X_train1.csv')
X_val = pd.read_csv('data/output/X_val1.csv')
y_train = pd.read_csv('data/output/y_train1.csv').values.ravel()
y_val = pd.read_csv('data/output/y_val1.csv').values.ravel()

# Define params for hyperparameter tuning
param_space = {
  'n_estimators': hp.choice('n_estimators', np.arange(700, 2400, 15, dtype=int)),
  'max_depth': hp.choice('max_depth', np.arange(50, 120, 10, dtype=int)),
  'max_features': hp.choice('max_features', np.arange(3, 7, 1, dtype=int)),
  'min_samples_split': hp.choice('min_samples_split', [2, 5, 10]),
  'min_samples_leaf': hp.choice('min_samples_leaf', [1, 2, 4, 5]),
  'bootstrap': hp.choice('bootstrap', [True, False])
}


def train_model(params):
    """Main function of module. Train machine learning model.

    Args:
        params (Integer/Boolean): The params to be validated with hyperparameter tuning.

    Returns:
        dict: Returns mean absolute error and status.
    """
    # Create and train model
    rf_model = RandomForestRegressor(**params, random_state=1234)
    rf_model.fit(X_train, y_train)

    # Evaluate Metrics
    rf_y_pred = rf_model.predict(X_val)
    rf_metrics = eval_model(y_val, rf_y_pred)
    mae = mean_absolute_error(y_val, rf_y_pred)

    # Log params, metrics and model to MLflow
    mlflow.log_params(params)
    mlflow.log_metrics(rf_metrics)

    # End run
    mlflow.end_run()

    return {"loss": mae, "status": STATUS_OK}

# Run hyperparameter tuning as mlflow runs
with mlflow.start_run() as run:
    best_params = fmin(
        fn=train_model,
        space=param_space,
        algo=tpe.suggest,
        max_evals=96,
        return_argmin=False)  # Return params, instead of indices

# Build final model with optimal parameters
final_model = RandomForestRegressor(**best_params, random_state=1234).fit(X_train, y_train)
mlflow.log_params(final_model.get_params())

# Evaluate final model
y_pred = final_model.predict(X_val)
metrics = eval_model(y_val, y_pred)

# Save final model
mlflow.log_metrics(metrics)
mlflow.sklearn.log_model(final_model, "final_model")
