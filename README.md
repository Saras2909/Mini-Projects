# 🚀 Mini Projects Repository

This repository is a collection of small, practical Python projects focused on machine learning, data science, and recommendation systems. Each project is self-contained and includes its own dataset, notebook, and documentation.

---

## 📌 Project Index

| Project | Category | Technologies | Description |
| :--- | :--- | :--- | :--- |
| [Boston](./Boston) | Machine Learning + API | Python, pandas, scikit-learn, FastAPI, joblib, statsmodels | Predicts Boston housing prices using linear regression and exposes a REST API for single and batch predictions. |
| [Logistic_Regresion](./Logistic_Regresion) | Classification | Python, pandas, scikit-learn, matplotlib | Predicts diabetes using logistic regression on a health dataset. |
| [Movie_Recommender](./Movie_Recommender) | Recommendation System | Python, pandas, NumPy, NLTK, scikit-learn | Recommends similar movies using content-based filtering and cosine similarity. |

---

## 📁 Repository Structure

```text
Mini_Projects/
├── README.md
├── Boston/
│   ├── Models/
│   │   ├── predictor.py
│   │   ├── Regression.ipynb
│   │   ├── boston_model.pkl
│   │   ├── boston_scaler.pkl
│   │   └── boston_features.pkl
│   ├── Src/
│   │   └── server.py
│   ├── .gitignore
│   └── readme.md
├── Logistic_Regresion/
│   ├── diabetes_prediction_dataset.csv
│   ├── logistic.ipynb
│   ├── readme.md
│   └── .gitignore (if present)
├── Movie_Recommender/
│   ├── Data/
│   │   ├── data_preproces.ipynb
│   │   ├── tmdb_5000_movies.csv
│   │   └── tmdb_5000_credits.csv
│   ├── Readme.md
│   └── .gitignore (if present)
└── .gitignore
```

---

## 🧠 Included Projects

### 1. Boston Housing Prediction
- Builds a linear regression model to estimate house prices.
- Uses feature selection and multicollinearity checks.
- Saves trained model artifacts for reuse.
- Runs a FastAPI service with endpoints for predicting a single value and uploading CSV files.

Project folder: [Boston](./Boston)

### 2. Diabetes Prediction with Logistic Regression
- Uses a medical dataset to classify whether a patient has diabetes.
- Applies preprocessing and train/test splitting.
- Evaluates classification performance using metrics and visualizations.

Project folder: [Logistic_Regresion](./Logistic_Regresion)

### 3. Movie Recommender System
- Uses the TMDB movie dataset to generate movie recommendations.
- Builds a content-based recommendation engine using tags and cosine similarity.
- Recommends similar movies for a selected title.

Project folder: [Movie_Recommender](./Movie_Recommender)

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone <your-repository-url>
cd Mini_Projects
```

### 2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3. Install dependencies
Install the required libraries for the project you want to run:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels fastapi uvicorn pydantic joblib nltk
```

---

## ▶️ Run Each Project

### Boston
```bash
cd Boston
python Models/predictor.py
uvicorn Src.server:app --reload
```
Then open:
- http://127.0.0.1:8000/docs

### Logistic Regression
```bash
cd Logistic_Regresion
jupyter notebook logistic.ipynb
```

### Movie Recommender
```bash
cd Movie_Recommender
jupyter notebook Data/data_preproces.ipynb
```

---

## 🛠️ Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Statsmodels
- FastAPI
- Uvicorn
- NLTK

---

## ✅ Summary

This repository is designed for learning and experimenting with real-world ML and data science workflows. It combines model training, evaluation, API deployment, and recommendation logic in a compact and beginner-friendly format.

Each project has its own detailed documentation in its respective folder, so you can explore them individually.

---

## 📚 Notes

- Some projects depend on external datasets and may require internet access to download them.
- It is recommended to use a dedicated virtual environment for each project.
- These are educational projects intended for practice, experimentation, and learning.

