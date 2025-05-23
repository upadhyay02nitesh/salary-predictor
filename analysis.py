import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

data=pd.read_csv('salary_prediction_with_issues.csv')
print(data.info())  # Display information about the dataset
print("Data loaded successfully.")
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.show()



print(data[data.isnull().any(axis=1)])  # Display rows with missing values
print(data.isnull().sum())


numeric_cols = data.select_dtypes(include=['number']).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())


print(data.isnull().sum())

# Check for duplicate rows
# print("Duplicate rows in the dataset: ", data.duplicated().sum())

data.drop_duplicates(inplace=True)  # Remove duplicate rows

print(data.info())  # Display information about the dataset after removing duplicates

# sns.pairplot(data, diag_kind='kde')
# plt.show()

def remove_outliers_iqr(df, column):
    df[column] = pd.to_numeric(df[column], errors='coerce')
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

for col in ['Salary', 'Experience', 'Age']:
    data = remove_outliers_iqr(data, col)
# for col in data.columns:
#     q1 = data[col].quantile(0.25)
#     q3 = data[col].quantile(0.75)
#     iqr = q3 - q1
#     lower_bound = q1 - 1.5 * iqr
#     upper_bound = q3 + 1.5 * iqr
#     outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
#     # print(f"Outliers in {col}:")
#     print(outliers)
    # print(data.info())
    # print(data.info())
data.to_csv("cleaned_salary_prediction.csv", index=False)
print("Data loaded successfully.")
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.show()
x=data.iloc[:,:-1]
y=data["Salary"]
# print(x)
# print(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print("Data split into training and testing sets.")

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
print("Linear regression model trained.")
print(lr.predict([[17,38]])[0])
print(f"Model Accuracy: {lr.score(x_test, y_test) * 100:.2f}%")


# coefficients = lr.coef_      # These are the weights for each feature
# intercept = lr.intercept_    # This is the bias (constant term)
# print(coefficients)
# print(intercept)

# print("Model Formula:")
# print(f"Salary = {coefficients[0]:.2f} * Experience + {coefficients[1]:.2f} * Age + {intercept:.2f}")