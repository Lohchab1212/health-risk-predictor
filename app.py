from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Laod Models and Features
model = joblib.load('diabetes_model.pkl')
feature_cols = joblib.load('feature_cols.pkl')

# Create a FastAPI app

app = FastAPI(title = "Diabetes Risk predcitor",description="Predcits diabetes risk based on health indicators",version="1.0")

# Define input data structure
class HealthData(BaseModel):
    HighBP: float
    HighChol: float
    CholCheck: float
    BMI: float
    Smoker: float
    Stroke: float
    HeartDiseaseorAttack: float
    PhysActivity: float
    Fruits: float
    Veggies: float
    HvyAlcoholConsump: float
    AnyHealthcare: float
    NoDocbcCost: float
    GenHlth: float
    MentHlth: float
    PhysHlth: float
    DiffWalk: float
    Sex: float
    Age: float
    Education: float
    Income: float

@app.get("/")
def home():
    return {"message":"diabetes Risk Predcitor API is running!"}

# Prediction route
@app.post("/predict")
def predict(data:HealthData):
    input_dict = data.dict()
    input_df = pd.DataFrame([input_dict])

    input_df['BMI_Age'] = input_df['BMI']*input_df['Age']
    input_df['BMI_BP'] = input_df['BMI']*input_df['HighBP']

    input_df = input_df[feature_cols]
#Make Prediction
    prediciton = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    return{"prediction":int(prediciton),
           "risk":"High Risk" if prediciton ==1 else "Low Risk",
           "probability":round(float(probability),3),
           "message":"Please consult a doctor for proper diagnosis"
           }