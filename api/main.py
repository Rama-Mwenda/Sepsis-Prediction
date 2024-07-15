#main.py
from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

app = FastAPI() #instantiate FastAPI

#load ml models and encoder
forest = joblib.load("../models/RandomForest.joblib")
ada = joblib.load("../models/Adaboost.joblib")
logreg = joblib.load("../models/LogReg.joblib")
encoder = joblib.load("../models/encoder.joblib")


#define input features
class input_features(BaseModel):
    plasma: float
    bt1: float
    pressure: float
    bt2: float
    bt3: float
    bmi: float
    bt4: float
    age: float
    insurance: str


#create endpoints
@app.get("/")
def index():
    return ({'Status':'API is online'})

@app.post("/rfpredict")
def random_forest_predict(data:input_features):
    df = pd.DataFrame([data.model_dump()])
    prediction = forest.predict(df)
    pred_int = int(prediction[0])
    prediction_label = encoder.inverse_transform([pred_int])[0]
    probability = round(float(forest.predict_proba(df)[0][pred_int] * 100), 2)
    results = {"prediction": prediction_label, "probability": probability}
    return {"results": results}


@app.post("/adapredict")
def adaboost_predict(data:input_features):
    df = pd.DataFrame([data.model_dump()])
    prediction = ada.predict(df)
    pred_int = int(prediction[0])
    prediction_label = encoder.inverse_transform([pred_int])[0]
    probability = round(float(ada.predict_proba(df)[0][pred_int] * 100), 2)
    results = {"prediction": prediction_label, "probability": probability}
    return {"results": results}


@app.post("/lrpredict")
def logreg_predict(data:input_features):
    df = pd.DataFrame([data.model_dump()])
    prediction = logreg.predict(df)
    pred_int = int(prediction[0])
    prediction_label = encoder.inverse_transform([pred_int])[0]
    probability = round(float(logreg.predict_proba(df)[0][pred_int] * 100), 2)
    results = {"prediction": prediction_label, "probability": probability}
    return {"results": results}

