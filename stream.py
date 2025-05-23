import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Salary Predictor", layout="wide")
st.title("📈 Salary Prediction Using Linear Regression")

data=pd.read_csv("cleaned_salary_prediction.csv")

st.header("🧮 Predict Salary")

# Features and target
x = data[['Experience', 'Age']]
y = data['Salary']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(x_train, y_train)
accuracy = model.score(x_test, y_test) * 100

st.metric(label="🎯 Model Accuracy", value=f"{accuracy:.2f}%")

exp = st.number_input("Enter Experience (years)", min_value=0.0, max_value=50.0, step=0.5)
age = st.number_input("Enter Age", min_value=18, max_value=70, step=1)

if st.button("Predict Salary"):
    pred_salary = model.predict([[exp, age]])[0]
    st.success(f"💰 Predicted Salary: ₹{pred_salary:,.2f}")
    coefficients = model.coef_      # These are the weights for each feature
    intercept = model.intercept_  
    
    st.write(f"Salary = {coefficients[0]:.2f} * Experience + {coefficients[1]:.2f} * Age + {intercept:.2f}")

    st.write("Salary = m1 * x1 + m2 * x2 + b")
    st.write("Where:")
    st.write("m1 = Coefficient of Experience")
    st.write("m2 = Coefficient of Age")
    st.write("x1 = Experience")
    st.write("x2 = Age")
    st.write("b = Intercept")
    
