# 📉 Customer Churn Prediction (Simulated Data)

This project explores how to build and evaluate machine learning models for predicting customer churn, using synthetic datasets tailored to reflect various real-world churn scenarios. It is part of a strategic portfolio demonstrating applied data science skills for business impact.

## 🧠 Goal

To simulate and analyze churn prediction models under different class imbalance settings:

- 🟡 **Moderately imbalanced churn** (~20%)
- 🟠 **Highly imbalanced churn** (<10%)
- 🔴 **Rare event churn** (<5%)

## 🧪 What’s Included

- Simulated customer activity data
- ML models tailored to each churn scenario
- Model evaluation using:
  - ROC and PR curves
  - Confusion matrices
  - Threshold tuning for rare events

## ✅ Techniques Used

- Logistic Regression with class weights
- SMOTE oversampling for minority class
- XGBoost with scale adjustment
- Evaluation via `scikit-learn` metrics and visualizations

## 🛠️ Tech Stack

- **Python 3.x**
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn`
- `xgboost`
- `imblearn` (for SMOTE)

## 📦 How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```


📌 This is a work in progress. Folder structure and final organization will evolve as the project grows.

