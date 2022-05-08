# This file is a sample template for flask app.py which can be deployed on server for making predictions.

# import libraries
from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)  # create flask app
model = joblib.load('model.pkl')  # Loading pkl model


@app.route("/")  # at address / i.e, homepage
def home():
    # this is homepage of application which would be deployed
    return render_template('index.html')


@app.route('/predict', methods=['POST'])  # at address /predict
def predict():
    # receiving the input values from HTML form.
    int_features = [x for x in request.form.values()]
    # instead of this, we can also upload csv file directly and read data using read_csv and fed it to model.
    # converting feature inputs in supportable format for model
    final_features = [np.array(int_features)]
    # predicting mostly matched canonical dealership & respective probability
    prediction, probability = model.predict(final_features)
    # printing decisions
    return render_template('index.html',
                           prediction_text=f'Given dealership matches\
                                 with {prediction} with probability {probability}')


if __name__ == "__main__":
    # run application
    app.run()
