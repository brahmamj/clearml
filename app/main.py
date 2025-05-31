import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from typing import Any, List, Optional, Union
import datetime
from pydantic import BaseModel
import os
import joblib
from fastapi import APIRouter, HTTPException, Body, FastAPI, Request, Response
from fastapi.encoders import jsonable_encoder
import pandas as pd
import numpy as np
import json
import prometheus_client as prom
from sklearn.metrics import r2_score
from prometheus_fastapi_instrumentator import Instrumentator

relative_path = './model/xgboost_model.pkl'
absolute_path = os.path.abspath(relative_path)
model =  joblib.load(absolute_path)

app = FastAPI(title="Patient -Survival- Prediction")
root_router = APIRouter()


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    #predictions: Optional[List[int]]
    predictions: Optional[int]


class DataInputSchema(BaseModel):
    age: Optional[int]
    anaemia: Optional[int]
    creatinine_phosphokinase: Optional[int]
    diabetes: Optional[int]
    ejection_fraction: Optional[int]
    high_blood_pressure: Optional[int]
    platelets: Optional[int]
    serum_creatinine: Optional[float]
    serum_sodium: Optional[float]
    sex: Optional[int]
    smoking: Optional[int]
    time: Optional[float]
    


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]

@root_router.get("/")
def index(request: Request) -> Any:
    """Basic HTML response."""
    body = (
        "<html>"
        "<body style='padding: 10px;'>"
        "<h1>Welcome to the API</h1>"
        "<div>"
        "Check the docs: <a href='/docs'>here</a>"
        "</div>"
        "</body>"
        "</html>"
    )

    return HTMLResponse(content=body)



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@root_router.get("/test")
def test_route():
    return {"message": "Test route is working"}

example_input = {
    "inputs": [
        {
           'age':55,
           'anaemia':0,
           'creatinine_phosphokinase':7861,
           'diabetes':0,
           'ejection_fraction':38,
           'high_blood_pressure':0,
           'platelets':38,
           'serum_creatinine':1.1,
           'serum_sodium':136.0,
           'sex':1,
           'smoking':0,
           'time':6.0
           }
    ]
}

@root_router.post("/predict", response_model=PredictionResults, status_code=200)
async def predict(input_data: MultipleDataInputs = Body(..., example=example_input)) -> Any:
    """
    Bike rental count prediction with the bikeshare_model
    """

    input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))
    
    pred = int(model.predict(input_df)[0])
    #results = "There are chances for DEATH_EVENT" if pred ==1 else "There are no chances for DEATH EVENT"

    #if results["errors"] is not None:
        #raise HTTPException(status_code=400, detail=json.loads(results["errors"]))

    return PredictionResults(
        errors=None,
        version="1.0",
        predictions=pred,
    )
r2_metric = prom.Gauge('patient_survial_r2_score', 'R2 score for random 100 test samples')
def update_metrics():
    """
    Update metrics 
    """
    data = pd.read_csv(os.path.abspath("./dataset/heart_failure_clinical_records_dataset.csv"))
    data = data.sample(100)
    data_feat = data.drop('DEATH_EVENT', axis=1)
    data_target = data['DEATH_EVENT'].values
    r2 = r2_score(data_target, model.predict(data_feat))
    r2_metric.set(r2)

@root_router.get("/metrics")
def metrics():
    """
    Get metrics
    """
    update_metrics()
    return Response(media_type="text/plain", content= prom.generate_latest())  

instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

app.include_router(root_router)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 