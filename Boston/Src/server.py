from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from fastapi.responses import StreamingResponse
import joblib
import pandas as pd
import io
import pathlib

app = FastAPI(title="Boston House Price Prediction API")

# Resolve directory and load model files directly from the Models folder
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "Models"

house_model = joblib.load(MODEL_DIR / "boston_model.pkl")
features = joblib.load(MODEL_DIR / "boston_features.pkl")

try:
    scaler = joblib.load(MODEL_DIR / "boston_scaler.pkl")
except Exception:
    scaler = None


# Schema for the input
class Housefeatures(BaseModel):
    crim: float = Field(..., ge=0, description="Per capita crime rate by town")
    zn: float = Field(..., ge=0, description="Proportion of residential land zoned for lots over 25,000 sq.ft")
    indus: float = Field(..., ge=0, description="Proportion of non-retail business acres per town")
    chas: float = Field(..., ge=0, le=1, description="Charles River dummy variable (1 if tract bounds river; 0 otherwise)")
    nox: float = Field(..., gt=0, description="Nitric oxides concentration (parts per 10 million)")
    rm: float = Field(..., gt=0, description="Average number of rooms per dwelling")
    age: float = Field(..., ge=0, description="Proportion of owner-occupied units built prior to 1940")
    dis: float = Field(..., gt=0, description="Weighted distances to five Boston employment centres")
    rad: float = Field(..., ge=1, description="Index of accessibility to radial highways")
    ptratio: float = Field(..., gt=0, description="Pupil-teacher ratio by town")
    b: float = Field(..., ge=0, description="1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town")
    lstat: float = Field(..., ge=0, description="% lower status of the population")


# home
@app.get("/")
def home():
    return {
        "message": "Boston House Price Prediction is working.",
        "status": "Running",
        "predict": "to predict go to /predict",
        "docs": "/docs"
    }


# model info
@app.get("/model")
def get_model_info():
    return {
        "status": "running",
        "model": "linear regression",
        "features": features,
        "avg_error": "3300"
    }


# predict
@app.post("/predict")
def predict(Features: Housefeatures):
    try:
        house_data = Features.model_dump()
        input_data = pd.DataFrame([house_data])[features]

        if scaler is not None:
            input_scaled = pd.DataFrame(scaler.transform(input_data), columns=features)
            predicted = house_model.predict(input_scaled)
        else:
            predicted = house_model.predict(input_data)


        price = float(predicted[0] * 1000)
        avg_error = 3300
        return {
            "Price in usd": f"${price:,.0f}",
            "Confidence in range": f"${max(0, price - avg_error):,.0f} to ${(price + avg_error):,.0f}"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction Failed Due To : {str(e)}"
        )


@app.post("/predict_file")
async def predict_file(file: UploadFile = File(...)):  # Upload file is an object, async means python can work on something else while it waits
    if file.filename.endswith(".csv"):
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))  # bytes io is used to read the binary content of the file into pandas read_csv
        required_features = features
        missing_feature = [
            feature for feature in required_features if feature not in df.columns  # Columns which are required by model but not in the input file
        ]
        extra_feature = [
            feature for feature in df.columns if feature not in required_features  # Extra features present in the input file which are not required by model
        ]
        if missing_feature:
            raise HTTPException(
                status_code=400,
                detail=f"Missing features: {', '.join(missing_feature)} not in input file"
            )
        if extra_feature:
            raise HTTPException(
                status_code=400,
                detail=f"Extra features: {', '.join(extra_feature)}"
            )
        if len(df) == 0:
            raise HTTPException(
                status_code=400,
                detail="Input file is empty"
            )
        try:
            input_data = df[features]
            if scaler is not None:
                input_scaled = pd.DataFrame(scaler.transform(input_data), columns=features)
                prediction = house_model.predict(input_scaled)
            else:
                prediction = house_model.predict(input_data)


            df["Predicted Price"] = (prediction * 1000)
            df["Predicted Price"] = df["Predicted Price"].apply(lambda x: f"${x:,.0f}")
            output = df.to_csv(index=False)
            return StreamingResponse(
                io.StringIO(output),
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=predictions.csv"}
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Prediction Failed Due To : {str(e)}"
            )
    else:
        raise HTTPException(
            status_code=400,
            detail="Input file must be a CSV file"
        )
