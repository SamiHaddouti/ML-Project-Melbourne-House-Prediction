"""
Data processing for building a random forest machine learning algorithm.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from helper_functions import identify_class, remove_outliers

URL = 'https://raw.githubusercontent.com/SamiHaddouti/Machine-Learning-Project/main/data/melb_data.csv'

melb_df = pd.read_csv(URL)

# Rename columns
melb_df.rename({'Lattitude': 'Latitude', 'Longtitude': 'Longitude'}, axis=1, inplace=True)

# Remove outliers
df_filtered = remove_outliers(melb_df, 'BuildingArea', 0.05, 0.05)
df_filtered2 = remove_outliers(df_filtered, 'Landsize', 0.10, 0.05)   # and removing most outliers
df_filtered3 = remove_outliers(df_filtered2, 'YearBuilt', 0.001, 0) # remove extremely old outlier

# Only keep relevant features (that show correlation with price)
df_selected = df_filtered3[['Price', 'Rooms', 'YearBuilt', 'Suburb', 'Bathroom',\
                            'Distance', 'Latitude', 'Longitude', 'BuildingArea', 'Landsize']]
# building area und landsize raus
# Car rein

# Drop Na(N)
cleaned_df = df_selected.dropna()

# Feature Engineering

# Houses <= 1960 -> historic / houses > 1960 -> modern
cleaned_df['HouseAgeType'] = cleaned_df['YearBuilt']\
                                .apply(lambda x: 'historic' if x <= 1960 else 'modern')

# Dividing suburbs into three classes
suburb_mean_price = cleaned_df.groupby('Suburb')['Price'].mean()

# Add/map mean suburb prices to suburbs in df
cleaned_df['suburb_mean_price'] = cleaned_df['Suburb'].map(suburb_mean_price)

cleaned_df['SuburbClass'] = cleaned_df['suburb_mean_price']\
                                .apply(lambda x: identify_class(x, suburb_mean_price))

# Check for false suburb classes (0)
checked_df = cleaned_df[cleaned_df.SuburbClass != 0]

select_df = checked_df[['Price', 'SuburbClass', 'Rooms', 'HouseAgeType', 'Bathroom',\
                       'Distance', 'Latitude', 'Longitude', 'BuildingArea', 'Landsize']]
# BuildingArea und Landsize raus
# car rein

# Encode categorical columns (HouseAgeType)
melb_df_hot_encoded = pd.get_dummies(select_df)

# Save final df
final_df = melb_df_hot_encoded
final_df.to_csv('data/output/final_df1.csv', index=False)

# Set X and y for train test split
X = final_df.drop(columns='Price')
y = final_df['Price']

# Split data into train, val, test
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, 
                                                  random_state=1234, shuffle=True)

X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                                                  test_size=0.15, random_state=1234)

# Save data sets as csv
X_train.to_csv('data/output/X_train1.csv', index=False)
X_val.to_csv('data/output/X_val1.csv', index=False)
X_test.to_csv('data/output/X_test1.csv', index=False)
y_train.to_csv('data/output/y_train1.csv', index=False)
y_val.to_csv('data/output/y_val1.csv', index=False)
y_test.to_csv('data/output/y_test1.csv', index=False)