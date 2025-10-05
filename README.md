# Heart Disease Prediction: Machine Learning Project 
Complete ML pipeline for binary classification to predict heart disease. Includes data preprocessing, model comparison (Logistic Regression, RF), hyperparameter tuning, and feature importance analysis.

## Project Overview
This project applies machine learning classification algorithms to predict the presence of heart disease using the UCI Heart Disease Dataset. The pipeline includes extensive data preprocessing, model comparison, hyperparameter tuning, and detailed performance analysis.

## Key Outcomes

| Metric | Score |
| :--- | :--- |
| **Final Model** | Tuned Random Forest Classifier |
| **ROC-AUC** | 0.9096 (Excellent discriminatory power) |
| **Accuracy** | 0.8333 |
| **F1-Score** | 0.8077 |

---

## Top Predictive Features (Feature Importance)

The model identified the most influential factors for heart disease prediction:
1.  **Thal\_Reversable Defect:** The most important factor, indicating insufficient blood flow during stress.
2.  **Oldpeak:** ST depression from exercise relative to rest.
3.  **Thal\_Normal:** Indicating the absence of a defect.
4.  **Thalch:** Maximum heart rate achieved.

## Repository Contents
- **`Disease_Prediction.py`**: The complete Python script containing the ML pipeline.
- **`heart_disease_data.csv`**: The dataset used for training and testing.
- **`plots/`**: Contains the generated visualizations.
    - `feature_importance.png` 
    - `tuned_random_forest_confusion_matrix.png`
