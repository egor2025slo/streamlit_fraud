# 🛡️ FraudGuard AI: Credit Card Fraud Detection System

An end-to-end Machine Learning solution designed to detect fraudulent transactions in highly imbalanced financial data. The project focuses on **minimizing financial losses** while maintaining a high user experience by reducing false positives.

## 📌 Business Problem
*   **Context:** Fraudulent transactions represent only **0.17%** of the data.
*   **Challenge:** Standard models prioritize accuracy (99.9%), failing to catch fraud. Aggressive models block too many genuine users (High False Positives).
*   **Goal:** Balance **Precision** and **Recall** to minimize financial loss and customer churn.

## 🛠️ Solution Architecture
1.  **Preprocessing:** 
    *   `RobustScaler` to handle extreme outliers in transaction amounts.
    *   **SMOTE** applied within a Cross-Validation pipeline to prevent Data Leakage.
2.  **Model:** 
    *   **XGBoost Classifier** tuned with **Optuna**.
    *   Optimized for **AUPRC** (Area Under Precision-Recall Curve).
3.  **Explainability:** 
    *   **SHAP** analysis revealed that behavioral features (`V14`, `V4`) are stronger indicators of fraud than the transaction amount itself.
4.  **Deployment:** 
    *   Interactive Web App built with **Streamlit**.
    *   Implements a "Traffic Light" risk system (Approved / Manual Review / Blocked).

## 📊 Key Results
*   **AUPRC:** 0.85 (High reliability).
*   **Recall:** 0.86 (Catches 86% of fraud).
*   **Precision:** 0.75 (Only 1 false alarm for every 3 genuine blocks).
*   **Business Impact:** Threshold tuning reduced False Positives by **~80%** compared to the baseline, saving operational costs for the call center.

## 🚀 How to Run
1. Clone the repository:
   git clone https://github.com/YOUR_USERNAME/streamlit_fraud.git