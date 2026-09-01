# Boston Housing Price Predictor

A Linear Regression machine learning project and FastAPI service designed to predict housing prices (`medv`) using the Boston Housing Dataset while addressing multicollinearity, scaling features, and serving inference endpoints.

## Dataset
URL: https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv

---

## Project Workflow

1. **Data Ingestion & Filtering**
   - Loads the dataset and separates target variable `medv` from feature matrix `X`.
   - Calculates correlation matrix with threshold of `0.85` to drop collinear features (e.g. `tax`).
   - Preserves removed features in a dedicated `Removed_Features` DataFrame.

2. **Feature Scaling & VIF Analysis**
   - Standardizes remaining features using `StandardScaler`.
   - Evaluates Variance Inflation Factor (VIF) and statsmodels OLS summary.

3. **Model Training & Artifact Serialization**
   - Fits a `LinearRegression` model on scaled training features.
   - Serializes `boston_model.pkl`, `boston_scaler.pkl`, and `boston_features.pkl` directly inside the `Models/` directory using `joblib`.

4. **FastAPI Web Service (`Src/server.py`)**
   - Serves prediction REST API endpoints powered by FastAPI and Pydantic validation.

---

## Project Structure

```
Boston/
├── Models/
│   ├── predictor.py         # Main script for data processing, training & serialization
│   ├── Regression.ipynb     # Jupyter Notebook for exploratory analysis
│   ├── boston_model.pkl     # Trained Linear Regression model
│   ├── boston_scaler.pkl    # Fitted StandardScaler transformer
│   └── boston_features.pkl  # List of selected feature names
├── Src/
│   └── server.py            # FastAPI application entry point
├── .gitignore               # Git ignore rules
└── readme.md                # Project documentation
```

---

## How to Run

### 1. Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels fastapi uvicorn pydantic joblib
```

### 2. Train and Save Model Artifacts
Run the predictor script to process data, train the model, and save `.pkl` files in `Models/`:
```bash
python Models/predictor.py
```

### 3. Start the FastAPI Server
Launch the API server using Uvicorn from the project root:
```bash
uvicorn Src.server:app --reload
```

---

## API Endpoints

- **`GET /`**: Health check and welcome message.
- **`GET /model`**: Returns model specifications, selected features list, and average error margin.
- **`POST /predict`**: Accepts JSON payload with feature values and returns estimated house price in USD.
- **`POST /predict_file`**: Accepts CSV file upload with house samples and returns downloadable CSV with predicted prices.
- **Interactive Documentation**: Access OpenAPI docs at `http://127.0.0.1:8000/docs`.
