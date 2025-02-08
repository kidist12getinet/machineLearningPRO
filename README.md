#Stock Price Prediction - Machine Learning Project

Overview

This project is a machine learning-based stock price prediction system that forecasts the closing stock price based on historical financial data.
The model is implemented using XGBoost and deployed using FastAPI for real-time predictions.

Features

Data Preprocessing: Cleans and scales stock market data

Machine Learning Model: Uses XGBoost for regression-based prediction

API Deployment: Provides real-time stock price predictions via FastAPI

Web Interface: A simple UI for user interaction

Files and Directories

main.py - FastAPI backend for the model prediction API

index.html - Frontend user interface for input and prediction display

requirements.txt - Dependencies required to run the project

xgboost_model.pkl - Trained XGBoost model

standard_scaler.pkl - Preprocessing scaler for data normalization

stock.CSV - Dataset used for model training

stockpro.ipynb - Jupyter Notebook containing exploratory data analysis and model training steps

Installation and Setup

#1. Clone the Repository

git clone [https://github.com/kidist12getinet/machineLearningPRO.git]
cd stock-price-prediction

#2. Install Dependencies

Ensure you have Python 3.8+ installed. Then, install the required packages:

pip install -r requirements.txt

#3. Run the FastAPI Server

uvicorn main:app --host 0.0.0.0 --port 8000 --reload

#4. Access the Web UI

Once the server is running, open your browser and go to:

http://127.0.0.1:8000/

Enter stock values and get predictions.

API Endpoints

1. Root Endpoint (GET /************************************************************************ )

Returns the frontend HTML page.

2. Prediction Endpoint (POST /predict************************************************************************ )

Takes stock data as input and returns the predicted closing price.

Example Request:

{
    "Open": 150.25,
    "High": 155.00,
    "Low": 149.50,
    "Adj_Close": 152.00,
    "Volume": 1000000
}

Example Response:

{
    "predicted_close": 152.67
}

Future Improvements

Implement LSTM-based deep learning models for better accuracy

Integrate real-time stock market data API for dynamic predictions

Deploy on AWS Lambda or Google Cloud Functions

License

This project is for educational purposes only. Use it responsibly.

#Author

Developed by: K/Mariam G.
For inquiries, contact: [kidistgetinet072@gmail.com]

