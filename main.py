from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import numpy as np
import datetime
import joblib

model = joblib.load(open("models/Car_Price_Regressor.pkl", "rb"))
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/predictor", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("predictor.html", {"request": request})


@app.post("/predictor/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    model_year: int = Form(...),
    present_price: float = Form(...),
    kms_driven: int = Form(...),
    owner: int = Form(...),
    fuel_type: str = Form(...),
    seller_type: str = Form(...),
    transmission_type: str = Form(...),
):
    year = datetime.date.today().year - model_year
    if fuel_type.lower() == "petrol":
        fuel = 239
    elif fuel_type.lower() == "diesel":
        fuel = 60
    else:
        fuel = 2

    seller = 106 if seller_type == "individual" else 195
    transmission = 261 if transmission_type == "manual" else 40

    input_list = [present_price, kms_driven, fuel, seller, transmission, owner, year]
    final_features = [np.array(input_list)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)
    return templates.TemplateResponse(
        "predictor.html", context={"request": request, "prediction": output}
    )
