
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import uvicorn

# Load the pipeline
with open("pipeline_v1.bin", "rb") as f:
    pipeline = pickle.load(f)

# Create FastAPI app
app = FastAPI()

class Lead(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

@app.post("/predict")
def predict(lead: Lead):
    # Convert to dictionary
    lead_dict = lead.dict()

    # Make prediction
    probability = pipeline.predict_proba([lead_dict])[0][1]

    return {"probability": probability}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
