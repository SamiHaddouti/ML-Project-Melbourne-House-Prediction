"""
Helper functions for rd_forest.py.
"""

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def remove_outliers(df, column, q_low, q_high):
    """Function to remove outliers from a DataFrame column.

    Args:
        df (DataFrame): The DataFrame with the specified column.
        column (String): The column to be filtered in DataFrame.
        q_low (Float): The lower quantile param serves as bottom filter for values.
        q_high (Float): The upper quantile param serves as top filter for values.

    Returns:
        DataFrame: The filtered DataFrame.
    """
    min_q = df[f'{column}'].quantile(q_low)
    max_q = df[f'{column}'].quantile(1-q_high)

    return df[(df[f'{column}'] < max_q) & (df[f'{column}'] > min_q)]


def identify_class(avg_price, suburb_mean_price):
    """Function to distinguish suburbs from melbourne data set into three classes based on
    third quantiles.

    Args:
        x (Float): Values/prices from cleaned_df['suburb_mean_price'].
        suburb_mean_price (Float): The mean property price for an suburb.

    Returns:
        Integer: Returns class as integer based on the quantile evaluation.
    """
    one_third_quantile = suburb_mean_price.quantile(0.33)
    two_third_quantile = suburb_mean_price.quantile(0.66)
    suburb_class = 0

    if avg_price >= two_third_quantile:
        suburb_class = 1
    if one_third_quantile <= avg_price < two_third_quantile:
        suburb_class = 2
    if avg_price < two_third_quantile:
        suburb_class = 3

    return suburb_class

def eval_model(y_val, y_pred):
    """Function to calculate and evaluate the metrics of an model.

    Args:
        y_val (list): Data set to validate the predictions of a model.
        y_pred (array): Predictions of an model.

    Returns:
        dict: Returns metrics (mae, mse, rmse, r2) from evaluation of model.
    """
    mae = mean_absolute_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)

    metrics = {
        "mae" : mae,
        "mse" : mse,
        "rmse" : rmse,
        "r2" : r2
        }

    return metrics
