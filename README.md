# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
~~~
1.Import the libraries and read the data frame using pandas.
2.Calculate the null values present in the dataset and apply label encoder.
3.Determine test and training data set and apply decison tree regression in dataset.
4.Calculate Mean square error,data prediction and r2. 
~~~
## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: konduru santhosh
RegisterNumber: 212225240074 
*/

# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# -------------------------------
# Read CSV File
# -------------------------------
df = pd.read_csv("employee.csv")


# -------------------------------
# Features and Target
# -------------------------------
X = df[['Experience', 'Age', 'Rating']]  # Independent variables
y = df['Salary']                          # Target variable

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Train Decision Tree Regressor
# -------------------------------
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# Prediction on Test Set
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# Evaluation Metrics
# -------------------------------
print("\n--- Model Evaluation ---")
print("Mean Absolute Error (MAE) :", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error (MSE)  :", mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error    :", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score                  :", r2_score(y_test, y_pred))

# -------------------------------
# Prediction for New Employee
# -------------------------------
# Example: Experience=5, Age=30, Rating=4
new_employee = [[5, 30, 4]]
predicted_salary = model.predict(new_employee)
print("\nPredicted Salary for new employee:", predicted_salary[0])


```

## Output:

#Data head:
<br>
<img width="482" height="322" alt="image" src="https://github.com/user-attachments/assets/216c6106-4cc4-4e0c-8895-5a8e804d2cac" />
<br>
#Data info:
<br>
<img width="745" height="289" alt="image" src="https://github.com/user-attachments/assets/da1a7d1d-d054-45f5-be74-e75755a4d270" />
<br>
#isnull() sum():
<br>
<img width="239" height="106" alt="image" src="https://github.com/user-attachments/assets/7a7d0b4d-9eee-4756-a82a-48d5d62afc32" />
<br>
#Daata head for salary:
<br>
<img width="401" height="290" alt="image" src="https://github.com/user-attachments/assets/abbf12e6-fe73-48ed-98b7-ac021f6059ae" />
<br>
#mean squared error:
<br>
<img width="194" height="37" alt="image" src="https://github.com/user-attachments/assets/74f0dcc9-81e3-40d5-b66e-186ed63a71b4" />
<br>
#r2 value:
<br>
<img width="319" height="38" alt="image" src="https://github.com/user-attachments/assets/6e36f320-d192-46f5-967a-e6af5111d55f" />



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
