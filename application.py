from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
model = pickle.load(open("LinearRegressionModel.pkl", "rb"))
car = pd.read_csv("cleaned.csv")

@app.route("/")
def index():
    car_models = sorted(car["name"].unique())
    companies = sorted(car["company"].unique())
    year = sorted(car["year"].unique(), reverse = True)
    fuel = sorted(car["fuel"].unique())
    transmission = sorted(car["transmission"].unique())
    owner = sorted(car["owner"].unique())
    companies.insert(0,"Select Company")
    return render_template("index.html", companies = companies, car_models = car_models, years = year, fuels = fuel, transmissions = transmission, owners = owner)

@app.route('/home.html')
def home():
    return render_template("home.html")

@app.route("/predict", methods = ["POST"])
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_model')
    year = int(request.form.get('year'))
    fuel = request.form.get('fuel')
    transmission = request.form.get('transmission')
    owner = request.form.get('owner')
    km_driven = int(request.form.get('km_driven'))
    print(company,car_model,year,fuel,transmission,owner,km_driven)

    prediction = model.predict(pd.DataFrame({'name': [car_model],'company': [company],'year': [year],'fuel': [fuel],'transmission': [transmission],'owner': [owner],'km_driven': [km_driven]}))
    print(prediction)

    return str(np.round(prediction[0], 2))

if __name__ == "__main__":
    app.run(debug = True)