from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import xgboost as xgb
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
# Load the trained model and scaler
model = joblib.load('xgboost_model.pkl')  # Update with your model filename
scaler = joblib.load('standard_scaler.pkl')   # Update with your scaler filename

# Create FastAPI instance
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
# Define the input data model
class StockData(BaseModel):
    Open: float
    High: float
    Low: float
    Adj_Close: float
    Volume: float

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the prediction endpoint
@app.post("/predict")
def predict(stock_data: StockData):
    # Convert input data to DataFrame
    data = pd.DataFrame([{
        "Open": stock_data.Open,
        "High": stock_data.High,
        "Low": stock_data.Low,
        "Adj Close": stock_data.Adj_Close,
        "Volume": stock_data.Volume
    }])
    
    # Scale the input data
    data_scaled = scaler.transform(data)
    
    # Make predictions
    prediction = model.predict(data_scaled)
    predicted_value = float(prediction[0])
    return {"predicted_close": predicted_value}

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")