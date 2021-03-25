#This is Heroku Deployment Lectre
from flask import Flask, request, render_template
import os
import pickle

print(os.getcwd())
path = os.getcwd()

with open('Models/LogisticRegressionModel.pkl', 'rb') as f:
    logistic = pickle.load(f)

with open('Models/RandomForestModel.pkl', 'rb') as f:
    randomforest = pickle.load(f)

with open('Models/SVMModel.pkl', 'rb') as f:
    svm_model = pickle.load(f)


def get_predictions(Age, Sex, Chest_Pain, Resting_Blood_Pressure, Serum_Cholestoral,
                    Fasting_Blood_Sugar, Resting_ECG_Results, Maximum_HeartRate,
                    Exercise_Induced_Angina, STDepression_Induced, req_model):

    mylist = [Age, Sex, Chest_Pain, Resting_Blood_Pressure, Serum_Cholestoral,
              Fasting_Blood_Sugar, Resting_ECG_Results, Maximum_HeartRate,
              Exercise_Induced_Angina, STDepression_Induced, req_model]

    mylist = [float(i) for i in mylist]
    vals = [mylist]

    if req_model == 'Logistic':
        #print(req_model)
        return logistic.predict(vals)[0]

    elif req_model == 'RandomForest':
        #print(req_model)
        return randomforest.predict(vals)[0]

    elif req_model == 'SVM':
        #print(req_model)
        return svm_model.predict(vals)[0]
    else:
        return "Cannot Predict"


app = Flask(__name__)


@app.route('/')
def homepage():
    return render_template('home.html')


@app.route('/', methods=['POST', 'GET'])
def my_form_post():
    if request.method == 'POST':
        Age = request.form['Age']
        Sex = request.form['Sex']
        Chest_Pain = request.form["Chest_Pain"]
        Resting_Blood_Pressure = request.form["Resting_Blood_Pressure"]
        Serum_Cholestoral = request.form["Serum_Cholestoral"]
        Fasting_Blood_Sugar = request.form["Fasting_Blood_Sugar"]
        Resting_ECG_Results = request.form["Resting_ECG_Results"]
        Maximum_HeartRate = request.form["Maximum_HeartRate"]
        Exercise_Induced_Angina = request.form["Exercise_Induced_Angina"]
        STDepression_Induced = request.form["STDepression_Induced"]
        req_model = request.form['req_model']

        target = get_predictions(Age, Sex, Chest_Pain, Resting_Blood_Pressure, Serum_Cholestoral,
                                 Fasting_Blood_Sugar, Resting_ECG_Results, Maximum_HeartRate,
                                 Exercise_Induced_Angina, STDepression_Induced, req_model)

        if target==1:
            sickness_making = 'Customer is likely to have Cardiac Disease'
        else:
            sickness_making = 'Customer is unlikely to have Cardiac Disease'

        return render_template('home.html', target = target, sale_making = sickness_making)
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(debug=True)