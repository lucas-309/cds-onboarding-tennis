# Import statements

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os



# Remove NaN functions

# Given a dataframe, removes all rows that have NaN data.
def remove_nan_rows(df):
    return df.dropna(axis = 0)

# Given a dataframe, replace all NaNs with a summary statistic of your choice.
def replace_nan_with_mean(df):
    df_copy = df.copy()
    for col in df_copy.columns:
        if df_copy[col].isna().any():
            df_copy[col].fillna(df_copy[col].mean(), inplace=True)
    return df_copy

# Given a column of categorical data and an exhaustive list of labels, returns a one-hot encoding.
def one_hot_encode(column, labels):
    return pd.get_dummies(column, prefix=column.name).reindex(columns=[f"{column.name}_{label}" for label in labels], fill_value=0)


# Import Data set and clean 

df = pd.read_csv('atp_matches_2024.csv')
# Replace NaN values with mean
df = df.drop(columns=['winner_seed', 'winner_entry', 'loser_seed', 'loser_entry'])
df = remove_nan_rows(df)
print(df.columns)
print(df.shape)
df.head()

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.min_rows', None)

# Summary Statistics

def summarize(df):
    df = remove_nan_rows(df)

    # Separate numerical and categorical columns
    numerical_columns = df.select_dtypes(include=['number']).columns
    categorical_columns = df.select_dtypes(exclude=['number']).columns

    # Summary statistics for numerical columns
    # numerical_summary = df[numerical_columns].describe().T

    # Summary statistics for categorical columns (Unique values & value counts)
    # categorical_summary = {col: df[col].value_counts() for col in categorical_columns}

    # Display categorical summaries
    # for col, counts in categorical_summary.items():
    #     print(f"\nSummary for {col}:")
    #     #print(f"Unique values: {df[col].nunique()}")
    #     print(counts)

    # Separate numerical and categorical columns
    numerical_columns = df.select_dtypes(include=['number']).columns
    categorical_columns = df.select_dtypes(exclude=['number']).columns


    # Create mean DataFrames for each surface
    mean_hard = df[df['surface'] == 'Hard'][numerical_columns].mean().round(2)
    mean_clay = df[df['surface'] == 'Clay'][numerical_columns].mean().round(2)
    mean_grass = df[df['surface'] == 'Grass'][numerical_columns].mean().round(2)
    
    # Combine into one DataFrame
    mean_df = pd.concat(
        [mean_hard, mean_clay, mean_grass], 
        axis=1, 
        keys=['Hard', 'Clay', 'Grass']
    )

    # Compute variance column: sum of 3 pairwise squared differences
    variance = (
        (mean_df['Hard'] - mean_df['Clay']) ** 2 +
        (mean_df['Clay'] - mean_df['Grass']) ** 2 +
        (mean_df['Grass'] - mean_df['Hard']) ** 2
    ).round(4)

    mean_df['Variance'] = variance

    # Display the combined DataFrame
    display(mean_df)
    
    

summarize(df)



# Create a folder to save plots
plot_dir = "plots"
os.makedirs(plot_dir, exist_ok=True)

def plotting(df):
    # Histogram for each numerical column
    numerical_cols = ['winner_age', 'w_bpSaved', 'winner_rank_points', 'w_ace', 'w_svpt', 'l_df']
    for col in numerical_cols:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[col], bins=30, kde=True)
        plt.title(f'Histogram of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.savefig(f'{plot_dir}/histogram_{col}.png')
        plt.show()

    # Bar graphs for each categorical column
    categorical_cols = ['winner_name', 'loser_name']
    # I tried the other categorical columns, they don't make sense for a bar graph
    for col in categorical_cols:
        plt.figure(figsize=(10, 6))
        top_30 = df[col].value_counts().nlargest(30).index
        sns.countplot(y=df[col], order=top_30, palette='viridis')
        plt.title(f'Bar Graph of {col}')
        plt.xlabel('Count')
        plt.ylabel(col)
        plt.savefig(f'{plot_dir}/bargraph_{col}.png')
        plt.show()
        plt.close()

    # Violin plots for selected pairs (categorical vs numerical)
    violin_categorical = ['winner_name', 'loser_name', 'tourney_name']
    violin_numerical = ['winner_age', 'loser_age', 'winner_rank_points']

    for cat_col, num_col in zip(violin_categorical, violin_numerical):
        top_10_categories = df[cat_col].value_counts().nlargest(10).index  # Get top 30 categories
        subset_df = df[df[cat_col].isin(top_10_categories)]  # Filter dataset to only include these categories

        plt.figure(figsize=(12, 6))
        sns.violinplot(data=subset_df, x=cat_col, y=num_col, palette="coolwarm")
        plt.title(f'Violin Plot of {num_col} by {cat_col} (Top 10 Categories)')
        plt.xticks(rotation=45)
        plt.xlabel(cat_col)
        plt.ylabel(num_col)
        plt.show()
        plt.savefig(f'{plot_dir}/violin_{num_col}_by_{cat_col}.png')
        plt.close()

    # Scatter plots for selected numerical pairs
    selected_num_pairs = [('winner_age', 'winner_rank_points'), ('loser_age', 'winner_rank_points'), ('w_svpt', 'loser_rank_points')]

    for x_col, y_col in selected_num_pairs:
        plt.figure(figsize=(8, 6))
        top_x_values = df[x_col]
        top_y_values = df[y_col]
        subset_df = df[(df[x_col].isin(top_x_values)) & (df[y_col].isin(top_y_values))]
        sns.scatterplot(x=subset_df[x_col], y=subset_df[y_col], alpha=0.5)
        plt.title(f'Scatter Plot: {x_col} vs {y_col}')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.savefig(f'{plot_dir}/scatter_{x_col}_vs_{y_col}.png')
        plt.show()
        plt.close()
    

    # Correlation heatmap
    plt.figure(figsize=(12, 8))
    corr_matrix = df[numerical_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.savefig(f'{plot_dir}/correlation_heatmap.png')
    plt.show()
    plt.close()

    # Additional Plots
    # Plot 1: Box plot of Winner Age across Tournaments
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df['tourney_name'].value_counts().nlargest(30).index, y=df['winner_age'].value_counts().nlargest(30).index)
    plt.xticks(rotation=90)
    plt.title('Box Plot of Winner Age across Tournaments')
    plt.savefig(f'{plot_dir}/boxplot_age_by_tourney.png')
    plt.show()
    plt.close()

    # Plot 2: KDE plot of age across rank points
    plt.figure(figsize=(10, 6))
    sns.kdeplot(x=df['winner_age'].value_counts().nlargest(30).index, y=df['winner_rank_points'].value_counts().nlargest(30).index, cmap="Reds", fill=True)
    plt.title('Density Plot of ATP Tennis Players')
    plt.savefig(f'{plot_dir}/kde_tennis_players.png')
    plt.show()
    plt.close()

plotting(df)

# test that runs data visualization

plotting(df)

