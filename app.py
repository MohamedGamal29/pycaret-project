# data_loader.py
import pandas as pd

def load_data(file_path):
    return pd.read_csv(file_path)

# eda.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def perform_eda(data):
    # Display basic information about the dataset
    print("Dataset Information:")
    print(data.info())

    # Display summary statistics of numerical features
    print("\nSummary Statistics:")
    print(data.describe())

    # Visualize the distribution of the target variable
    plt.figure(figsize=(8, 5))
    sns.countplot(x='target', data=data)
    plt.title('Distribution of Target Variable')
    plt.show()

    # Visualize the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.show()

    # Pairplot to visualize relationships between numerical features
    sns.pairplot(data, hue='target')
    plt.suptitle('Pairplot of Numerical Features by Target', y=1.02)
    plt.show()

    # Boxplot to identify outliers in numerical features
    numerical_cols = data.select_dtypes(include='number').columns
    plt.figure(figsize=(12, 6))
    for i, col in enumerate(numerical_cols, 1):
        plt.subplot(2, 2, i)
        sns.boxplot(x='target', y=col, data=data)
        plt.title(f'Boxplot of {col} by Target')
    plt.tight_layout()
    plt.show()

# app.py
import streamlit as st
from data_loader import load_data
from eda import perform_eda

def main():
    st.title('Your Machine Learning Assistant')

    # Upload file through Streamlit
    uploaded_file = st.file_uploader("Choose a file", type=['csv'])

    if uploaded_file is not None:
        # Load data
        data = load_data(uploaded_file)

        # Perform EDA
        perform_eda(data)

        # Display loaded data
        st.write('Loaded Data:')
        st.write(data)

# Run the Streamlit app
if __name__ == '__main__':
    main()
