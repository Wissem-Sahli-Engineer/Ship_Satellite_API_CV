# pyrefly: ignore [missing-import]
from fastapi import FastAPI
# pyrefly: ignore [missing-import]
from tf_keras.models import load_model
# pyrefly: ignore [missing-import]
import numpy as np
from utils import format



# Load the model
model = load_model("models/model_3/keras_Model.h5", compile=False)

# Load the labels
class_names = open("models/model_3/labels.txt", "r").readlines()

# Define the app
app = FastAPI()

# terminal run :
"""
source .venv/bin/activate && uvicorn main:app --reload
"""

@app.get("/")
def read_root():
    return {"message": "Ship Detection API is up and running!"}

@app.post('/classify')
def classify(img_url):
    
    image = format(img_url)

    # Predicts the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return {'Prediction ' : class_name[2:-1],
            'Confidence score ': float(confidence_score)
            }