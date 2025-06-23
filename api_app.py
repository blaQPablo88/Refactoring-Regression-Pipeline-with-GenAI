# app.py
from fastapi import FastAPI
import pandas as pd
from src.preprocessing import clean_infinite_values, standard_scale
from src.feature_selection import select_features
from joblib import load

app = FastAPI()

# Load saved model and transformer
model = load("models/LinearRegression.joblib")
transformer = load("models/scaler_transformer.joblib")
selected_columns = load("models/selected_columns.joblib")

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    df = clean_infinite_values(df)
    df = df[selected_columns]
    df_scaled = transformer.transform(df)
    prediction = model.predict(df_scaled)
    return {"prediction": prediction.tolist()}
