from fastapi import FastAPI, Request
from app.modelProcess import ModelProcess, DataShape, Gender
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles


# Initialize
app = FastAPI()
dpr = ModelProcess()
templates = Jinja2Templates(directory="app/templates")
# app.mount("app/static", StaticFiles(directory="static"), name="static")

@app.get("/", include_in_schema=False)   # path operation decorator: ex: accessing "/" with GET method
def index(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.html", context={}
    )


@app.post("/predict")
async def predictFraud(data: DataShape):
    result = dpr.predict(data) 
    return {
        "is_fraud": bool(result),
        "confidence": 0.99,
        "transaction_id": f"TXN-{data.step}Z99"
    }