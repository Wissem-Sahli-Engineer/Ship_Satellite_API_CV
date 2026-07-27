# pyrefly: ignore [missing-import]
from fastapi import FastAPI


app = FastAPI()

@app.post('/classify')
def classify(img_url):
    pass 