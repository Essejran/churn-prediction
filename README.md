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

📌 This is a work in progress. Folder structure and final organization will evolve as the project grows.s

## 📦 How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```


#### Choice Of Base Model & Evaluation Metrics:

#### ✅ Churn Model Metrics
Choose metrics based on current priority

| Metric                                    | Why It's Useful                                                     | When to Use                                                                         |
| ----------------------------------------- | ------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| **Recall (Sensitivity)**                  | Measures how many actual churners you caught                        | Most important when false negatives are costly (missing a churner = lost customer)  |
| **Precision**                             | Measures how many of the predicted churners really churned          | Important if actions are expensive (e.g., offering discounts only to real churners, not annoying loyal customers with unnecessary retention actions) |
| **F1 Score**                              | Harmonic mean of precision and recall                               | Useful when you want a balance between precision and recall                         |
| **ROC AUC Score**                         | Measures how well the model ranks positives over negatives          | Great for overall model evaluation and comparison                                   |
| **PR AUC Score** *(Precision-Recall AUC)* | Better than ROC AUC in highly imbalanced settings                   | Especially useful when positive class (churn) is rare                               |
| **Confusion Matrix**                      | Gives raw counts of TP, FP, FN, TN                                  | Useful for stakeholder communication                                                |
| **Lift at K / Top-K Recall**              | Tells you how many churners you caught in the top-scoring customers | Great for marketing prioritization (e.g., top 10% most likely churners)             |


❌ __Accuracy:__ Misleading in imbalanced datasets. If 90% don’t churn in learning data, a model predicting “no churn” for everyone will still be 90% accurate — but useless.


##### 📖 Churn Model Metrics Interpretation

| Metric               | What It Means                                       | Good Sign                          | Bad Sign                          | Special Notes for Churn                   |
| -------------------- | --------------------------------------------------- | ---------------------------------- | --------------------------------- | ----------------------------------------- |
| **Accuracy**         | % of all predictions that are correct               | >90%                               | Often misleading                  | Ignore this for churn (imbalanced data)   |
| **Precision**        | Of all predicted churners, how many really churned? | High = you don’t annoy loyal users | Low = unnecessary retention costs | Important if retention actions are costly |
| **Recall**           | Of all real churners, how many did you catch?       | High = saves more customers        | Low = lost churners               | Most important in churn prevention        |
| **F1 Score**         | Balance of precision & recall                       | High = good tradeoff               | Low = biased model                | Use when both precision & recall matter   |
| **ROC AUC**          | How well does model rank churners vs. non-churners? | Close to 1.0                       | 0.5 = random guessing             | Good general performance metric           |
| **PR AUC**           | Area under Precision-Recall curve                   | High = effective for rare churn    | Low = poor signal on rare events  | Better than ROC AUC for rare churn        |
| **Confusion Matrix** | Raw count of TP, FP, FN, TN                         | -                                  | -                                 | Shows real-world mistakes                 |
| **Lift / Top-K**     | Not in this code, but useful for marketing          | -                                  | -                                 | Can be added later                        |
  
     
        
---  
  
##### 🔥 How Rare Is "Rare"? Hold My Stake
| Churn Rate | Model                                         | Technique                     | Key Metric                |
| ---------- | --------------------------------------------- | ----------------------------- | ------------------------- |
| **< 20%**  | `LogisticRegression(class_weight='balanced')` | Balanced class weights        | ROC AUC + F1              |
| **< 10%**  | `SMOTE + LogisticRegression`                  | Oversampling + scaling        | PR AUC + F1               |
| **< 5%**   | `SMOTE + XGBClassifier(scale_pos_weight=...)` | Advanced model + class weight | PR AUC + threshold tuning |


#### 📊 Experiment Tracking with MLflow

To ensure reproducibility and transparency across scenarios, this project uses **MLflow** for logging model parameters, evaluation metrics, and model artifacts.

Each notebook run is tracked as a distinct experiment based on the churn scenario:

- `Moderately imbalanced churn` (Logistic Regression with class weights)
- `Highly imbalanced churn` (SMOTE + Logistic Regression)
- `Rare event churn` (SMOTE + XGBoost)

**Logged elements include:**

- Scenario metadata (`churn_rate`, `model_type`, etc.)
- Key metrics (`precision`, `recall`, `ROC AUC`, `PR AUC`)
- Input examples for reproducibility
- Model signature for schema validation

All models are logged using `mlflow.sklearn.log_model()` with associated metadata to enable:

- 🔁 Version control and comparisons
- 📦 Artifact tracking for future deployment
- 📈 Robust experiment monitoring

> 🔐 *Why this matters:* MLflow enables auditability and structured decision-making, especially important in high-impact churn modeling where model updates must be tracked over time.
