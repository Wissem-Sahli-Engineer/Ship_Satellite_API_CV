# pyrefly: ignore [missing-import]
from fastapi import FastAPI , HTTPException
# pyrefly: ignore [missing-import]
from tf_keras.models import load_model
# pyrefly: ignore [missing-import]
import numpy as np
from utils import format
# pyrefly: ignore [missing-import]
from pydantic import BaseModel
# pyrefly: ignore [missing-import]
from fastapi.middleware. cors import CORSMiddleware

class Image (BaseModel):
    image_url: str



# Load the model
model = load_model("models/model_3/keras_model.h5", compile=False)

# Load the labels
class_names = open("models/model_3/labels.txt", "r").readlines()

# Define the app
app = FastAPI(
    title="Ship Satellite CV API",
    description="Ship classification model API powered by Teachable Machine / Keras",
)

# terminal run :
"""
source .venv/bin/activate && uvicorn main:app --reload
"""

origins = ["*"]
app.add_middleware(CORSMiddleware, 
                    allow_origins=origins, 
                    allow_credentials=True, 
                    allow_methods=["*"], 
                    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Ship Detection API is up and running!"}

@app.post('/classify')
def classify(image : Image):
    try:

        img_url = image.image_url
        
        image = format(img_url)

        # Predicts the model
        prediction = model.predict(image)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        return {'Prediction ' : class_name[2:-1],
                'Confidence score ': float(confidence_score)
                }
    
    except Exception as e :
        raise HTTPException(
            status_code=400,
            detail=f"Failed to process or classify image: {str(e)}",
        )