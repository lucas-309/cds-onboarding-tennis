#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def remove_nan_rows(df):
    """Remove all rows with any NaN values."""
    return df.dropna(axis=0)

def replace_nan_with_mean(df):
    """Replace NaNs with column mean."""
    df_copy = df.copy()
    for col in df_copy.columns:
        if df_copy[col].isna().any():
            df_copy[col].fillna(df_copy[col].mean(), inplace=True)
    return df_copy

def label_encode(column):
    """Label-encode a pandas Series."""
    labels = column.unique()
    label_dict = {label: idx for idx, label in enumerate(labels)}
    return column.map(label_dict)

def train_evaluate_regressor(X_train, y_train, X_val, y_val, model_name):
    """Train a LinearRegression and return model, train MSE, and val MSE."""
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_train, y_train_pred, label='Training Data')
    plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], 'r--', label='Perfect Prediction')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{model_name} - Training vs Predicted')
    plt.legend()
    plt.show()

    train_mse = mean_squared_error(y_train, y_train_pred)
    val_mse = mean_squared_error(y_val, y_val_pred)
    return model, train_mse, val_mse

def main():
    # Load and initial clean
    df = pd.read_csv('atp_matches_2024.csv')
    df = df.drop(columns=['winner_seed', 'winner_entry', 'loser_seed', 'loser_entry'])
    df = remove_nan_rows(df)

    # Encode categorical columns
    categorical_columns = ["tourney_name", "winner_name", "loser_name", "surface", "winner_hand"]
    for name in categorical_columns:
        df[name + '_encoded'] = label_encode(df[name])

    # Subsample for speed
    df = df.sample(n=1000, random_state=42)

    # Split dataset
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    print(f"Training Set: {len(train_df)}")
    print(f"Validation Set: {len(valid_df)}")
    print(f"Test Set: {len(test_df)}")

    # Feature/target split for classification example
    target_col = 'tourney_name_encoded'
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_val = valid_df.drop(columns=[target_col])
    y_val = valid_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    # Correlation analysis
    numeric_df = train_df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    print(corr_matrix)
    
    # Scatterplots vs winner_rank_points
    for feature in numeric_df.columns:
        plt.figure(figsize=(6,4))
        sns.scatterplot(x=df[feature], y=df['winner_rank_points'])
        plt.title(f'{feature} vs winner_rank_points')
        plt.show()

    # Regression examples
    features_regressor_1 = ['winner_age', 'winner_rank_points']
    features_regressor_2 = ['loser_age', 'loser_rank_points']
    features_regressor_3 = ['draw_size', 'winner_rank']
    reg_target = 'winner_name_encoded'

    # Prepare data for regression
    X = df[features_regressor_1 + features_regressor_2 + features_regressor_3]
    y = df[reg_target]
    X_train_r, X_temp_r, y_train_r, y_temp_r = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val_r, X_test_r, y_val_r, y_test_r = train_test_split(X_temp_r, y_temp_r, test_size=0.5, random_state=42)

    # Train and evaluate regressors
    reg1, train_mse_1, val_mse_1 = train_evaluate_regressor(X_train_r, y_train_r, X_val_r, y_val_r, 'Regressor 1')
    reg2, train_mse_2, val_mse_2 = train_evaluate_regressor(X_train_r, y_train_r, X_val_r, y_val_r, 'Regressor 2')
    reg3, train_mse_3, val_mse_3 = train_evaluate_regressor(X_train_r, y_train_r, X_val_r, y_val_r, 'Regressor 3')

    # Compare errors
    error_table = pd.DataFrame({
        'Regressor': ['Regressor 1', 'Regressor 2', 'Regressor 3'],
        'Training MSE': [train_mse_1, train_mse_2, train_mse_3],
        'Validation MSE': [val_mse_1, val_mse_2, val_mse_3]
    })
    print(error_table)

    # Test best model
    best_idx = error_table['Validation MSE'].idxmin()
    best_model = [reg1, reg2, reg3][best_idx]
    y_test_pred = best_model.predict(X_test_r)
    test_mse = mean_squared_error(y_test_r, y_test_pred)
    print(f"Test MSE of Best Model: {test_mse}")

if __name__ == "__main__":
    main()
