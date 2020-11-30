from flask import Flask, request, render_template, redirect
import pickle
from os import path
import pandas as pd
import numpy as np

from utils.data_pre_process import data_processing
from utils.churn_prediction import train_model

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    model = request.form['model']

    acc_length = int(request.form['acc_length'])
    international_plan = int(request.form['international_plan'])
    voice_plan = int(request.form['voice_plan'])
    vmail_number = int(request.form['vmail_number'])
    day_minutes = int(request.form['day_minutes'])
    day_calls = int(request.form['day_calls'])
    eve_minutes = int(request.form['eve_minutes'])
    eve_calls = int(request.form['eve_calls'])
    night_minutes = int(request.form['night_minutes'])
    night_calls = int(request.form['night_calls'])
    intl_minutes = int(request.form['intl_minutes'])
    intl_calls = int(request.form['intl_calls'])
    cust_service_calls = int(request.form['cust_service_calls'])

    if path.exists('pickle/' + model + '.pkl'):
        model = pickle.load(open('pickle/' + model + '.pkl', 'rb'))
        array = np.array(
            [[acc_length, international_plan, voice_plan, vmail_number, day_minutes, day_calls, eve_minutes, eve_calls,
              night_minutes, night_calls, intl_minutes, intl_calls, cust_service_calls]])
        pred = model.predict(array)
        return render_template("result.html", result=pred)
    else:
        data_after_processing = data_processing()
        telecom = data_after_processing['un-scaled']['train']

        train_model(telecom)
        print('******************Completed******************')
    return render_template("index.html")




if __name__ == '__main__':
    app.run()
