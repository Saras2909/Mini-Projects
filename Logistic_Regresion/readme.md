# Diabetes Prediction using Logistic Regression

This mini-project demonstrates binary classification using **Logistic Regression** to predict whether a patient has diabetes based on medical diagnostic measurements.

---

## 📁 Project Structure

```text
Logistic_Regresion/
│
├── diabetes_prediction_dataset.csv   # Dataset with 100,000 patient records
├── logistic.ipynb                    # Jupyter Notebook containing code and visualizations
└── readme.md                         # Project documentation
```

---

## 📊 Dataset Overview

The dataset contains 100,000 rows and 9 columns:

- **gender**: Patient gender (Mapped: `Female: 0`, `Male: 1`, `Other: 2`)
- **age**: Patient age
- **hypertension**: 0 if patient doesn't have hypertension, 1 if patient has hypertension
- **heart_disease**: 0 if patient doesn't have heart disease, 1 if patient has heart disease
- **smoking_history**: Smoking history (Mapped: `never: 0`, `No Info: 1`, `current: 2`, `former: 3`, `ever: 4`, `not current: 5`)
- **bmi**: Body Mass Index
- **HbA1c_level**: Hemoglobin A1c level
- **blood_glucose_level**: Blood glucose level
- **diabetes** *(Target)*: 0 = No Diabetes, 1 = Diabetes

---

## ⚙️ Steps & Workflow

1. **Data Loading & Inspection**: Read dataset using Pandas and inspect data types and missing values.
2. **Feature Mapping**: Convert categorical variables (`gender` and `smoking_history`) into numerical values using mapping dictionaries.
3. **Train-Test Split**: Separate features (`X`) and target (`y`), then split the data (80% training, 20% testing).
4. **Model Training**: Train a `LogisticRegression` model from `scikit-learn`.
5. **Evaluation**: Calculate model accuracy (~95.06%) and generate predictions on test data.
6. **Visualization**: Plot a Confusion Matrix and Actual vs Predicted Probability scatter plot.

---

## 🚀 Requirements

To run the notebook, install the required Python libraries:

```bash
pip install pandas numpy matplotlib scikit-learn
```
Dataset Link:- https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset