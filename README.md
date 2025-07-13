# A Machine Learning Framework for Predicting Prolonged Hospitalization in Iranian Knee Joint Replacement Candidates

This project aims to develop a robust and interpretable machine learning pipeline to **predict prolonged length of hospital stay (LOS)** in Iranian patients undergoing **Total Knee Arthroplasty (TKA)**.

---

## 🎯 Project Goal

To assist clinicians and hospital administrators in identifying patients at risk of prolonged hospitalization after knee joint replacement, allowing for proactive planning and individualized post-op care.

---

## ⚙️ Pipeline Overview

### 1. **Preprocessing**
- **Function:** `encode_categorical_variables(df)`
- **File:** `preprocessing.py`
- **Purpose:** Converts raw mixed-type features (e.g., gender, comorbidities) into numeric one-hot encoded format for ML compatibility.
- **Output:** `encoded_dataset.csv`

### 2. **Train-Test Split & Scaling**
- **Libraries:** `train_test_split`, `StandardScaler`
- **Purpose:** Splits the dataset into stratified training and testing sets and standardizes features for optimal model performance.

### 3. **Model Training**
- **Class:** `ModelSelector`
- **File:** `model_maker.py`
- **Purpose:** Trains multiple ML classifiers (e.g., logistic regression, random forest, SVM, XGBoost) and saves the best-performing ones for evaluation.
- **Output:** Saved `.pkl` model files in `saved_models/`

### 4. **Visualization**
- **Class:** `ModelVisualizer`
- **File:** `visualizer.py`
- **Purpose:** Produces:
  - ROC curves for discrimination
  - Confusion matrices for performance overview
  - SHAP plots for model interpretability
- **Output:** PNG/HTML plots saved to `results/figures/`

### 5. **Evaluation**
- **Class:** `ModelEvaluator`
- **File:** `model_evaluator.py`
- **Purpose:** Calculates metrics such as:
  - Precision, Recall, F1-Score
  - Specificity, NPV
  - Log loss
- **Output:** `model_metrics.csv` with full evaluation summary

---

## 📊 Final Output

After running `mainRun.py` (or equivalent pipeline), the results include:
- A clean **encoded dataset**
- Multiple **trained ML models**
- Diagnostic plots: **ROC**, **SHAP**, **Confusion Matrix**
- A detailed **metrics table** (`model_metrics.csv`) summarizing all models

---

## 🧠 Requirements
- Python 3.8+
- See requirements.txt

---

## 📬 Questions?

For collaboration, clarification, or academic use, please reach out to the project author.
