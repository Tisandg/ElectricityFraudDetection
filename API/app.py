from fastapi import FastAPI, HTTPException
import pickle
import uvicorn
import numpy as np
from tsai.all import *
from tsai.inference import load_learner

app = FastAPI(
    title="Fraud electricity detection API",
    description="A simple FastAPI application to detect fraud electricity consumption",
    version="0.0.1",
    author="David Santiago Garcia Chicangana"
)


'''
Load the model and data when the API starts.
'''
@app.on_event("startup")
def load_model_and_data():
    global learner, data, dls, model, labels, splits, threshold
    model_path = './models/InceptionTime_v20_bestf1_learner.pkl'
    model = load_learner(model_path)
    threshold = 0.3

    with open('./data/data_triple_mask_normalized.pkl', 'rb') as data_file:
        data = pickle.load(data_file)
    
    with open('./data/labels.pkl', 'rb') as data_file:
        labels = pickle.load(data_file)

    with open('./data/splits.pkl', 'rb') as data_file:
        splits = pickle.load(data_file)

# Routes
@app.get("/")
async def root():
    return {"message": "Welcome to the Fraud Electricity Detection API"}

@app.post("/predict/")
def predict(data: dict):

    X_expanded = np.expand_dims(data['data'], axis=0)
    X_expanded.shape

    y_expanded = np.expand_dims(data['label'], axis=0)
    y_expanded.shape

    probas, target, preds = model.get_X_preds(X_expanded, y_expanded)
    
    true_label = data['label']
    prediction = preds
    probability = probas
    print(probability[0][1].item())

    prediction_adjusted = 0
    if probability[0][1].item() > 0.3:
        prediction_adjusted = 1
    
    return {
        "prediction": int(prediction[0]),
        "probability": probability.tolist(),
        "true_label": int(true_label),
        "prediction_adjusted": prediction_adjusted,
        "threshold": threshold
    }

@app.get("/predict/{record_id}")
def predict(record_id: int):

    # splits[0] # Train
    # splits[1] # Valid

    if record_id >= len(splits[1]):
        raise HTTPException(status_code=404, detail="Record not found")
    
    X_expanded = np.expand_dims(data[splits[1][record_id]], axis=0)
    X_expanded.shape

    y_expanded = np.expand_dims(labels[splits[1][record_id]], axis=0)
    y_expanded.shape

    probas, target, preds = model.get_X_preds(X_expanded, y_expanded)
    
    true_label = labels[splits[1][record_id]]
    prediction = preds
    probability = probas
    print(probability[0][1].item())

    prediction_adjusted = 0
    if probability[0][1].item() > 0.3:
        prediction_adjusted = 1
    
    return {
        "prediction": int(prediction[0]),
        "probability": probability.tolist(),
        "true_label": int(true_label),
        "prediction_adjusted": prediction_adjusted,
        "threshold": threshold
    }

if __name__ == "__main__":
        uvicorn.run(app, host="0.0.0.0", port=8000)