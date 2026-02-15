from fastapi import FastAPI, Request
from app.modelProcess import ModelProcess
from fastapi.templating import Jinja2Templates

# Initialize
app = FastAPI()
dpr = ModelProcess()
templates = Jinja2Templates(directory="app/templates")

@app.get("/", include_in_schema=False)   # path operation decorator: ex: accessing "/" with GET method
def index(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.html", context={}
    )