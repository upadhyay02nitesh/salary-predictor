import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Salary Predictor", layout="wide")
st.title("📈 Salary Prediction Using Linear Regression")

uploaded_file = st.file_uploader("Upload your salary dataset (CSV)", type=["csv"])

def remove_outliers_iqr(df, column):
    df[column] = pd.to_numeric(df[column], errors='coerce')
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("✅ Data loaded successfully!")

    # Pre-cleaning info
    st.subheader("📊 Before Cleaning Data Overview")
    st.write("Missing Value rows in the dataset: ",data.isnull().sum())
    st.write("Duplicate rows in the dataset: ", data.duplicated().sum())
    numeric_cols = data.select_dtypes(include=['number']).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
    data.drop_duplicates(inplace=True)

    st.subheader("📊 After Cleaning Data Overview")
    st.write("Missing Value rows in the dataset: ",data.isnull().sum().sum())
    st.write("Duplicate rows in the dataset: ", data.duplicated().sum())
    outlier_summary = []

    for col in ['Salary', 'Experience', 'Age']:
        data[col] = pd.to_numeric(data[col], errors='coerce')

        # Calculate bounds
        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Outliers before removal
        outliers_before = data[(data[col] < lower_bound) | (data[col] > upper_bound)]

        # Remove outliers
        data = remove_outliers_iqr(data, col)

        # Outliers after removal (ideally should be 0)
        outliers_after = data[(data[col] < lower_bound) | (data[col] > upper_bound)]

        outlier_summary.append({
            'Column': col,
            'Outliers Before': len(outliers_before),
            'Outliers After': len(outliers_after)
        })

    # Convert to DataFrame and display as a table
    outlier_df = pd.DataFrame(outlier_summary)
    st.subheader("📋 Outlier Summary Table")
    st.dataframe(outlier_df)


    # Split into two columns
    left_col, right_col = st.columns(2)

    with left_col:
        st.header("🧼 Data Cleanup & Analysis")

        st.subheader("🔍 Raw Data Preview")
        st.write(data.head())

        st.subheader("📊 Correlation Heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

        st.subheader("⬇️ Download Cleaned Data")
        cleaned_csv = data.to_csv(index=False).encode('utf-8')
        st.download_button("Download Cleaned CSV", cleaned_csv, "cleaned_salary_data.csv", "text/csv")

    