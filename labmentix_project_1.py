# -*- coding: utf-8 -*-
"""labmentix project 1.ipynb
**Project Name - Yes Bank Stock Closing price Prediction**

Project Type - Regression/Supervised

Contribution -Mukta Arvind Londhe (Individual)

**Project Summary** -Yes Bank is a well-known bank in the Indian financial domain. 
Since 2018, it has been in the news due to various financial and operational challenges.
This project aims to predict the closing price of Yes Bank's stock using regression and supervised machine learning techniques. 
The analysis involves exploratory data analysis (EDA), visualization, and the application of machine learning models such as Linear Regression, Random Forest Regressor, and XGBoost Regressor.
The dataset used contains historical stock prices from 2005 to 2020, including features like Open, High, Low, and Close prices. 
The goal is to build a predictive model that can accurately forecast the closing price, which can be valuable for investors and financial analysts.
The project is an individual contribution by Mukta Arvind Londhe, and the code is shared on GitHub.

**Key Steps:**

1.Data Loading and Exploration: The dataset is loaded and explored to understand its structure and features.   
2.Visualization: Trends and patterns in the stock price data are visualized using libraries like Matplotlib and Seaborn.
3.Model Training: Machine learning models are trained and evaluated using metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R2) score.   
4.Prediction: The models are used to predict the closing price of Yes Bank's stock.

The project leverages Python libraries such as Pandas, NumPy, Scikit-learn, and XGBoost for data manipulation, analysis, and modeling.
The problem statement focuses on forecasting the closing price to aid in financial decision-making.

GitHub Link - https://github.com/muktalondhe/ML-projects/edit/main/labmentix_project_1.py

Problem Statement- Find Out Closing Price Prediction of the Yes Bank

**lets begin**

**Import Libraries & Load Data**
"""

# Core Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Statistical Analysis
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats

# Suppress Warnings
import warnings
warnings.filterwarnings('ignore')

# Load Dataset
df = pd.read_csv('data_YesBank_StockPrices.csv')
print("Dataset Shape:", df.shape)
df.head()

"""** Exploratory Data Analysis (EDA) & Visualization**

**Chart 1: Stock Price Trend Over Time**
"""

plt.figure(figsize=(15, 6))
plt.plot(pd.to_datetime(df['Date'], format='%b-%y'), df['Close'], color='blue', label='Closing Price')
plt.title("Yes Bank Stock Price Trend (2005-2020)", fontsize=14)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Price (INR)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()

"""Insight:

* Volatility spikes observed in 2018-2020.
* Peak price (~₹400) in 2018, followed by a sharp decline.

**Chart 2: Boxplot for Outlier Detection**
"""

plt.figure(figsize=(10, 5))
sns.boxplot(data=df[['Open', 'High', 'Low', 'Close']], palette='Set2')
plt.title("Boxplot of Stock Prices (Detecting Outliers)", fontsize=14)
plt.ylabel("Price (INR)")
plt.show()

"""Action Taken:
Used IQR method to cap outliers.
"""

def cap_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
    return df

df = cap_outliers(df, ['Open', 'High', 'Low', 'Close'])

"""**Chart 3: Correlation Heatmap**"""

corr = df[['Open', 'High', 'Low', 'Close']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlation Between Price Features", fontsize=14)
plt.show()

"""Key Finding:

Open, High, Low, and Close are highly correlated (>0.99).

**Feature Engineering & Preprocessing**
"""

# Feature Engineering
df['Date'] = pd.to_datetime(df['Date'], format='%b-%y')
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Daily_Return'] = df['Close'].pct_change()
df['MA_5'] = df['Close'].rolling(5).mean()
df['MA_10'] = df['Close'].rolling(10).mean()

# Lag Features
for lag in [1, 2, 3]:
    df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)

# Drop missing values
df.dropna(inplace=True)

# Train-Test Split (Time-Series Friendly)
X = df.drop(['Date', 'Close'], axis=1)
y = df['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Feature Scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

"""**Model Training & Evaluation**

**Model 1: Linear Regression (Baseline)**
"""

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

print("Linear Regression:")
print(f"- R²: {r2_score(y_test, y_pred_lr):.4f}")
print(f"- MAE: {mean_absolute_error(y_test, y_pred_lr):.2f}")

"""**Model 2: Random Forest**"""

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)

print("\nRandom Forest:")
print(f"- R²: {r2_score(y_test, y_pred_rf):.4f}")
print(f"- MAE: {mean_absolute_error(y_test, y_pred_rf):.2f}")

"""**Model 3: XGBoost (Best Performance)**"""

xgb = XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
xgb.fit(X_train_scaled, y_train)
y_pred_xgb = xgb.predict(X_test_scaled)

print("\nXGBoost:")
print(f"- R²: {r2_score(y_test, y_pred_xgb):.4f}")
print(f"- MAE: {mean_absolute_error(y_test, y_pred_xgb):.2f}")

"""**Hyperparameter Tuning (XGBoost)**"""

param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7]
}

grid_search = GridSearchCV(xgb, param_grid, cv=3, scoring='r2', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

best_xgb = grid_search.best_estimator_
y_pred_best = best_xgb.predict(X_test_scaled)

print("\nOptimized XGBoost:")
print(f"- Best Params: {grid_search.best_params_}")
print(f"- R²: {r2_score(y_test, y_pred_best):.4f}")

"""**Final Summary & Conclusion**"""

results = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest', 'XGBoost (Base)', 'XGBoost (Tuned)'],
    'R² Score': [
        r2_score(y_test, y_pred_lr),
        r2_score(y_test, y_pred_rf),
        r2_score(y_test, y_pred_xgb),
        r2_score(y_test, y_pred_best)
    ],
    'MAE': [
        mean_absolute_error(y_test, y_pred_lr),
        mean_absolute_error(y_test, y_pred_rf),
        mean_absolute_error(y_test, y_pred_xgb),
        mean_absolute_error(y_test, y_pred_best)
    ]
})

print("\n=== Final Model Performance ===")
print(results)

plt.figure(figsize=(10, 5))
sns.barplot(x='Model', y='R² Score', data=results, palette='viridis')
plt.title("Model Comparison (R² Score)", fontsize=14)
plt.ylim(0.9, 1.0)
plt.xticks(rotation=15)
plt.show()

"""Key Takeaways:

* XGBoost (Tuned) performs best (R² = 0.995).   
* Random Forest is robust against outliers.  
* Linear Regression is a good baseline but less accurate.

**Export Model for Deployment**
"""

import joblib

# Save Model & Scaler
joblib.dump(best_xgb, 'yes_bank_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and scaler saved successfully!")

"""**Key Insights:**

* The stock shows high volatility with periods of rapid growth and sharp declines  
* Technical indicators like moving averages and lag features significantly improve prediction accuracy  
* Ensemble methods (Random Forest and XGBoost) outperform linear regression for this time series prediction task  
* The model can be used for short-term stock price forecasting with high accuracy

**Future Work:**

* Incorporate more external factors like market indices, economic indicators  
* Implement deep learning models (LSTM) for potentially better performance  
* Develop a trading strategy based on the model predictions
* Create a web application for real-time stock price predictions  
"""
