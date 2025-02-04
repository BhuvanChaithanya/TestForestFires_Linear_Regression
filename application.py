import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
application = Flask(__name__)
app = application

##import ridge regressor and standard scalar pickle

ridge_model = pickle.load(open('models/ridge.pkl','rb'))
standard_scaler = pickle.load(open('models/scaler_l2.pkl','rb'))


@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predictdata",methods= ["GET","POST"])
def predict_datapoint():
    if request.method == "POST":
        # Get values from form
        temperature = float(request.form.get("Temperature"))
        rh = float(request.form.get("RH"))
        ws = float(request.form.get("WS"))
        rain = float(request.form.get("Rain"))
        ffmc = float(request.form.get("FFMC"))
        dmc = float(request.form.get("DMC"))
        isi = float(request.form.get("ISI"))
        classes = float(request.form.get("Classes"))
        region = float(request.form.get("Region"))

        # Convert inputs to a NumPy array and reshape for the model
        input_data = np.array([[temperature, rh, ws, rain, ffmc, dmc, isi, classes, region]])
        
        # Apply standard scaling
        scaled_data = standard_scaler.transform(input_data)

        prediction = ridge_model.predict(scaled_data)

        return render_template('home.html', results =prediction[0])

    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0")