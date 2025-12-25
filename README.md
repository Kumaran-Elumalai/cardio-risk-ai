# CardioRiskAI

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification-orange.svg)
![Healthcare AI](https://img.shields.io/badge/Domain-Healthcare-green.svg)

## Overview
**CardioRiskAI** is a machine learning–based clinical intelligence system designed to classify myocardial infarction (MI) outcomes using structured patient data.

The project focuses on responsibly applying supervised learning to high-dimensional, imbalanced medical data in order to support early risk identification and outcome awareness.  
It demonstrates an end-to-end healthcare ML workflow — from domain-aware preprocessing to model deployment.

---

## Clinical Problem Statement
Myocardial Infarction (MI), commonly known as a heart attack, remains one of the leading causes of mortality worldwide, particularly within the first year following the event.

Accurate classification of MI outcomes can assist clinicians and healthcare systems in:
- Understanding patient risk profiles  
- Identifying high-risk cases early  
- Supporting data-driven clinical decision making  

**Objective:**  
Build a robust classification model to predict MI outcome categories (`LET_IS`) using patient-level clinical and demographic attributes.

---

## Dataset Overview
- **Records:** 1,700 patients  
- **Features:** 124 clinical, demographic, and diagnostic variables  
- **Target Variable:** `LET_IS` (MI outcome class)

The dataset includes variables related to:
- Age and gender  
- Blood pressure measurements  
- Cholesterol levels  
- Diabetes and hypertension indicators  
- Other clinically relevant attributes  

Due to the medical nature of the data, **domain awareness was applied throughout preprocessing and feature selection**.

---

## Methodology & System Design

### 1. Data Understanding & Quality Assessment
- Inspected dataset shape and structure  
- Analyzed missing values using heatmaps  
- Removed columns with excessive missingness **only after clinical relevance review**  
- Verified absence of duplicate records  

---

### 2. Missing Value Handling
Imputation strategies were selected based on feature type:
- **Binary features:** Mode  
- **Categorical features:** Mode  
- **Numerical features:** Mean  

Post-imputation validation confirmed no remaining null values.

---

### 3. Exploratory Data Analysis (EDA)
Key analyses included:
- Age distribution (majority between 50–70 years)  
- Gender distribution (male-dominant cohort)  
- Blood pressure and cholesterol trends  
- MI occurrence by:
  - Age
  - Gender
  - Diabetes status
  - Hypertension status  

Blood pressure categories (normal, elevated, stage 1–5 hypertension) were analyzed to understand clinical severity distribution.

---

### 4. Class Imbalance Handling
The target variable exhibited **significant class imbalance**, with the majority class dominating outcome distribution.

To address this:
- Applied **SMOTE (Synthetic Minority Over-sampling Technique)**
- Balanced the dataset prior to model training
- Ensured minority outcome classes were adequately represented

---

### 5. Model Development & Evaluation
Multiple classification models were trained and evaluated:
- Decision Tree Classifier  
- Random Forest Classifier  
- **XGBoost Classifier**

Model selection was based on **clinical relevance of metrics**, not accuracy alone:
- Precision
- Recall
- F1-score
- Support per class

---

### 6. Model Selection & Optimization
- **XGBoost** consistently outperformed other models across key metrics  
- Hyperparameter tuning performed using **GridSearchCV**  
- Final model achieved **~92% accuracy** with strong precision-recall balance  
- Cross-validation used to ensure generalization stability  

XGBoost was selected due to its ability to:
- Handle high-dimensional data  
- Capture non-linear relationships  
- Remain robust on imbalanced datasets  

---

## Results & Insights
- Class imbalance significantly affected baseline model performance  
- SMOTE improved recall for minority outcome classes  
- XGBoost provided the best trade-off between performance and interpretability  
- Feature importance analysis highlighted clinically meaningful predictors  

The final model demonstrates strong potential for **risk stratification support**, not automated diagnosis.

---

## Deployment
The trained model was deployed using **Streamlit**, enabling:
- Interactive input of patient attributes  
- Real-time MI outcome classification  
- Easy demonstration of model behavior  

Deployment logic is implemented in:
MCI_Web_App.py


This completes the **EDA → modeling → evaluation → deployment** lifecycle.

---

## Tech Stack
- **Language:** Python  
- **Data Processing:** Pandas, NumPy  
- **Visualization:** Matplotlib, Seaborn  
- **Modeling:** Scikit-learn, XGBoost  
- **Imbalance Handling:** imbalanced-learn (SMOTE)  
- **Evaluation Metrics:** Precision, Recall, F1-score  
- **Deployment:** Streamlit  

---

## Repository Structure
```
cardio-risk-ai/
│
├── MCI_EDA.ipynb
├── MCI_Feature_Selection_Model_Building_Evaluation_techniques.ipynb
├── MCI_Web_App.py
├── Myocardial_attribute.txt
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```


---

## Key Engineering Decisions
- Applied domain awareness during feature removal and imputation  
- Addressed class imbalance before model training  
- Prioritized recall and F1-score over raw accuracy  
- Selected XGBoost for performance stability on complex medical data  

---

## Learnings
- Importance of metric selection in healthcare ML  
- Risks of ignoring class imbalance in clinical datasets  
- Translating clinical intuition into data-driven features  
- Deploying ML responsibly for decision support use cases  

---

## Limitations & Ethical Considerations
- This model is **not a diagnostic tool**  
- Intended strictly for educational and research purposes  
- Predictions should not replace professional medical judgment  

---

## Future Improvements
- Incorporate SHAP-based explainability  
- Explore cost-sensitive learning approaches  
- Validate model on external clinical datasets  
- Add uncertainty estimation for predictions  

---

## License
This project is licensed under the **MIT License**.

---

## Author
**Kumaran Elumalai**  
AI / ML Engineer | Data Scientist  

🔗 GitHub: https://github.com/Kumaran-Elumalai  
🔗 LinkedIn: https://linkedin.com/in/kumaran-elumalai
