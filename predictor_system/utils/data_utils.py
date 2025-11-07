import pandas as pd
import os
import streamlit as st

def load_data(filename='employee_salary_data.csv'):
    """
    Load data from CSV file
    
    Args:
        filename (str): Path to the CSV file
        
    Returns:
        DataFrame or None: Loaded data or None if file doesn't exist
    """
    if os.path.exists(filename):
        return pd.read_csv(filename)
    return None

def save_data(df, filename='employee_salary_data.csv'):
    """
    Save DataFrame to CSV file
    
    Args:
        df (DataFrame): Data to save
        filename (str): Filename to save to
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        df.to_csv(filename, index=False)
        return True
    except Exception:
        return False

def append_data(new_df, filename='employee_salary_data.csv'):
    """
    Append new data to existing CSV file
    
    Args:
        new_df (DataFrame): New data to append
        filename (str): Filename to append to
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if os.path.exists(filename):
            # Load existing data
            existing_df = pd.read_csv(filename)
            # Append new data
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            # Save combined data
            combined_df.to_csv(filename, index=False)
        else:
            # If file doesn't exist, just save the new data
            new_df.to_csv(filename, index=False)
        return True
    except Exception:
        return False