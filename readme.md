# Boston Housing Price Predictor

A Linear Regression machine learning project designed to predict housing prices (`medv`) using the Boston Housing Dataset while addressing multicollinearity and evaluating feature importance.

# Dataset
URL: https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv

## Project Workflow

1. **Data Ingestion**
   - The dataset is loaded directly from a remote CSV source.
   - Features (`X`) and target variable (`medv` / `y`) are separated.

2. **Exploratory Data Analysis & Correlation Removal**
   - Computes an absolute correlation matrix and visualizes feature interactions using a Seaborn heatmap.
   - Identifies feature pairs exceeding a correlation threshold of `0.85` (e.g., `tax` and `rad`).
   - Preserves dropped features in a dedicated `Removed_Features` DataFrame to ensure no data is lost.
   - Drops highly correlated features from the main feature set to prevent multicollinearity.

3. **Feature Scaling & VIF Analysis**
   - Standardizes remaining features using `StandardScaler`.
   - Computes Variance Inflation Factor (VIF) and matrix condition number to verify multicollinearity reduction.

4. **Model Training & Evaluation**
   - Splits the data into 80% Training and 20% Testing sets.
   - Fits a `LinearRegression` model on the scaled features using scikit learn.
   - Evaluates model performance using $R^2$ Score, Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE).
   - Runs `statsmodels` OLS regression for statistical summary and coefficient p-values.

5. **Visualization**
   - Plots Actual vs. Predicted house prices for both Training and Test datasets.

## Project Structure

```
Boston/
├── Models/
│   ├── predictor.py         # Main script for data processing, model training & evaluation
│   └── Regression.ipynb     # Jupyter Notebook for exploratory analysis
├── server.py                # FastAPI entry point
├── .gitignore               # Ignored files (notebooks, cache, etc.)
└── readme.md                # Project documentation
```

## How to Run

### Prerequisites
Install the required dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels fastapi
```

### Execution
Run the main predictor script:
```bash
python Models/predictor.py
```

