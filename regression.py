"""
Build and train ridge regression machine learning algorithm.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from hyperopt import fmin, tpe, hp, STATUS_OK

import mlflow
import mlflow.sklearn

from helper_functions import identify_class, remove_outliers, eval_model

mlflow.set_experiment('DataExploration_Project - Ridge Regression')

URL = 'https://raw.githubusercontent.com/SamiHaddouti/Machine-Learning-Project/main/data/melb_data.csv'

melb_df = pd.read_csv(URL)

# Rename columns
melb_df.rename({'Lattitude': 'Latitude', 'Longtitude': 'Longitude'}, axis=1, inplace=True)

# Remove outliers
df_filtered = remove_outliers(melb_df, 'BuildingArea', 0.05, 0.05)
df_filtered2 = remove_outliers(df_filtered, 'Landsize', 0.10, 0.05)   # and removing most outliers
df_filtered3 = remove_outliers(df_filtered2, 'YearBuilt', 0.001, 0) # remove extremely old outlier

# Only keep relevant features (that show correlation with price)
df_selected = df_filtered3[['Price', 'Rooms', 'YearBuilt', 'Suburb', 'Bathroom', 'Car',\
                            'Distance', 'Latitude', 'Longitude']]

# Drop Na(N)
cleaned_df = df_selected.dropna()

# Feature Engineering

# Modern/historic houses
# houses <= 1960 -> historic / houses > 1960 -> modern
cleaned_df['HouseAgeType'] = cleaned_df['YearBuilt']\
                                .apply(lambda x: 'historic' if x <= 1960 else 'modern')

# Suburb Classes
# Dividing suburbs into three classes
suburb_mean_price = cleaned_df.groupby('Suburb')['Price'].mean()

# Add/map mean suburb prices to suburbs in df
cleaned_df['suburb_mean_price'] = cleaned_df['Suburb'].map(suburb_mean_price)

cleaned_df['SuburbClass'] = cleaned_df['suburb_mean_price']\
                                .apply(lambda x: identify_class(x, suburb_mean_price))

# Check for false suburb classes (0)
checked_df = cleaned_df[cleaned_df.SuburbClass != 0].reset_index()

final_df = checked_df[['SuburbClass', 'Price', 'Rooms', 'HouseAgeType', 'Bathroom', 'Car',\
                       'Distance', 'Latitude', 'Longitude']]

# Encode categorical columns (HouseAgeType)
melb_df_hot_encoded = pd.get_dummies(final_df)

X = melb_df_hot_encoded.drop(columns='Price')
y = melb_df_hot_encoded['Price']


# Split data into train, val, test
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1,
                                                            random_state=1234, shuffle=True)

X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                                                  test_size=0.15, random_state=1234)


param_space = {
    'alpha': hp.choice('alpha', [0.0001, 0.001, 0.01])
}

def train_model(params):
    # Create and train model.
    ridge_model = KernelRidge()
    #ridge_model = Ridge()
    ridge_model.fit(X_train, y_train)

    # Evaluate Metrics
    ridge_y_pred = ridge_model.predict(X_val)
    ridge_metrics = eval_model(y_val, ridge_y_pred)
    mae = mean_absolute_error(y_val, ridge_y_pred)

    # Log params, metrics and model to MLflow
    mlflow.log_params(params)
    mlflow.log_metrics(ridge_metrics)

    mlflow.end_run()
    return {"loss": mae, "status": STATUS_OK}


with mlflow.start_run() as run:
    best_params = fmin(
        fn=train_model, 
        space=param_space, 
        algo=tpe.suggest, # Tree of Parzen Estimator instead of random
        max_evals=32)