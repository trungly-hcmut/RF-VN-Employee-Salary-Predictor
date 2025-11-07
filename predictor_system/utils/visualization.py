import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def create_correlation_heatmap(df):
    """
    Create a correlation heatmap for numeric columns
    
    Args:
        df (DataFrame): Data to visualize
        
    Returns:
        matplotlib.figure.Figure: Correlation heatmap
    """
    # Create a copy of the dataframe with numeric columns only
    numeric_df = df.select_dtypes(include=['number'])  # Changed from np.number to 'number'
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Heatmap')
    
    return fig

def create_feature_importance_plot(feature_importance_df):
    """
    Create a bar plot of feature importance
    
    Args:
        feature_importance_df (DataFrame): DataFrame with Feature and Importance columns
        
    Returns:
        matplotlib.figure.Figure: Feature importance plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, ax=ax)
    ax.set_title('Feature Importance')
    
    return fig