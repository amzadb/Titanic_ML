from fastapi import FastAPI
import uvicorn

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model')))
from titanic_model import model, kmeans, predict_survival, summary_handler

# --- 2. SETUP THE "WAITER" (FastAPI) ---
app = FastAPI()

@app.get("/")
def home():
    return {"message": "Titanic Survival Prediction API is Online!"}

from fastapi import Query

@app.get("/summary")
def summary(prompt: str = Query(None, description="Summary prompt text")):
    return summary_handler(prompt)

@app.get("/predict")
def predict(
    pclass: int = Query(..., description="Passenger class (1, 2, or 3)"),
    sex: str = Query(..., description="Sex (male or female)"),
    age: float = Query(..., description="Age in years"),
    sibsp: int = Query(..., description="Number of siblings/spouses aboard"),
    parch: int = Query(..., description="Number of parents/children aboard"),
    fare: float = Query(..., description="Passenger fare"),
    embarked: str = Query(..., description="Port of Embarkation (C, Q, S)")
):
    return predict_survival(pclass, sex, age, sibsp, parch, fare, embarked)

# --- 3. RUN THE SERVER ---
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)