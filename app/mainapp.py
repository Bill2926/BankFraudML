from fastapi import FastAPI
from app.modelProcess import ModelProcess

# Initialize
app = FastAPI()
dpr = ModelProcess()

@app.get("/")   # path operation decorator: ex: accessing "/" with GET method
def index():
    print("DCMMMM")
    return {"message": "Success!"} # This sends data to the browser