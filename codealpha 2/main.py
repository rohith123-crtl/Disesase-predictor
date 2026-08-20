from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
from typing import List

# --- 1. Initialize the API ---
app = FastAPI(title="Disease Prediction API")

# --- 2. Configure CORS ---
# This allows your index.html file to communicate securely with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for development
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# --- 3. Load the Model and Symptoms ---
try:
    model = joblib.load('disease_model.pkl')
    symptoms_list = joblib.load('symptoms_list.pkl')
except Exception as e:
    raise RuntimeError(f"Error loading model or symptoms. Make sure you have run train_model.py first! Details: {e}")

# --- 4. Define Input Data Structure ---
class SymptomInput(BaseModel):
    symptoms: List[str]

# --- 5. Symptoms List Endpoint ---
@app.get("/symptoms")
def get_symptoms():
    return {"symptoms": list(symptoms_list)}

# --- 6. Create the Prediction Endpoint ---
@app.post("/predict")
def predict_disease(data: SymptomInput):
    # Check if the user actually sent any symptoms
    if not data.symptoms:
        raise HTTPException(status_code=400, detail="No symptoms provided")
    
    # Create an empty dataframe with zeros matching the model's expected columns
    input_df = pd.DataFrame(0, index=[0], columns=symptoms_list)
    
    # Set the provided symptoms to 1 in the dataframe
    for symptom in data.symptoms:
        if symptom in symptoms_list:
            input_df.at[0, symptom] = 1
            
    # Make the prediction using the loaded model
    prediction = model.predict(input_df)[0]
    
    # Return the result back to the frontend
    return {
        "predicted_disease": prediction, 
        "provided_symptoms": data.symptoms
    }