import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Data Cleaning and Visualization Model")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Original Data")
    st.write(df.head())
    
    # Handle missing values
    df_cleaned = df.copy()
    for col in df_cleaned.columns:
        if df_cleaned[col].dtype == "object":
            df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)
        else:
            df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)
    
    # Rename columns to proper format
    df_cleaned.columns = [col.strip().replace(" ", "_").lower() for col in df_cleaned.columns]
    
    st.write("### Cleaned Data")
    st.write(df_cleaned.head())
    
    # Data visualization
    st.write("### Data Visualization")
    
    # Bar Chart for categorical data
    categorical_cols = df_cleaned.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        st.write("#### Bar Charts")
        for col in categorical_cols:
            fig, ax = plt.subplots()
            df_cleaned[col].value_counts().plot(kind='bar', ax=ax)
            plt.title(f"Bar Chart of {col}")
            plt.xticks(rotation=45)
            st.pyplot(fig)
    
    # Box plot for numerical data
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        st.write("#### Box Plots")
        for col in numeric_cols:
            fig, ax = plt.subplots()
            sns.boxplot(y=df_cleaned[col], ax=ax)
            plt.title(f"Box Plot of {col}")
            st.pyplot(fig)
    
    # Histogram for numerical data
    if numeric_cols:
        st.write("#### Histograms")
        for col in numeric_cols:
            fig, ax = plt.subplots()
            df_cleaned[col].hist(ax=ax, bins=20)
            plt.title(f"Histogram of {col}")
            st.pyplot(fig)
    
    # Correlation Heatmap
    if len(numeric_cols) > 1:
        st.write("#### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df_cleaned[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        plt.title("Correlation Heatmap")
        st.pyplot(fig)
    
    st.success("Data cleaning and visualization completed successfully!")