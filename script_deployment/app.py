import pickle
import pandas as pd
from flask import Flask, request

app = Flask(__name__)

with open("model/model_kmean.pkl", "rb") as model_file:
    model_kmean = pickle.load(model_file)

LABEL = ["Type 1", "Type 2", "Type 3", "Type 4", "Type 5"]

@app.route("/")
def index():
    return {
        "status":"SUCCESS",
        "message":"Service is Up"
    }, 200

@app.route("/predict")
def predict():
    args = request.args
    fixed_acidity = args.get("fixed acidity", default=0, type=float)
    citric_acid = args.get("citric acid", default=0, type=float)
    residual_sugar = args.get("residual sugar", default=0, type=float)
    chlorides = args.get("chlorides", default=0, type=float)
    density = args.get("density", default=0, type=float)
    new_data = [[fixed_acidity, citric_acid, residual_sugar, chlorides, density]]
    new_data = pd.DataFrame(new_data, columns=["fixed acidity", "citric acid", "residual sugar", "chlorides", "density"])
    result = model_kmean.predict(new_data)
    result = LABEL[result[0]]
    return {
        "status":"SUCCESS",
        "input":{"Keasaman":fixed_acidity, "Asam sitrat":citric_acid, "Gula residual":
                 residual_sugar, "Klorida":chlorides, "Densitas":density},
        "result":result
    }, 200

app.run(debug=True)