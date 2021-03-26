#This is Heroku Deployment Lectre
from flask import Flask, request, render_template
import os
import pickle

print(os.getcwd())
path = os.getcwd()

with open('Models/LogisticRegressionMod.pkl', 'rb') as f:
    logistic = pickle.load(f)

with open('Models/SVMMod.pkl', 'rb') as f:
    svm_model = pickle.load(f)

with open('Models/RandomForestMod.pkl', 'rb') as f:
    randomforest = pickle.load(f)


def get_predictions(age, sex, chest_pain, resting_blood_pressure, serum_cholestoral, fasting_blood_sugar,
                    resting_ecg_results, maximum_heartrate, exercise_induced_angina, stdepression_induced,
                    slope, mayor_vessels,thal, req_model):

    mylist = [age, sex, chest_pain, resting_blood_pressure, serum_cholestoral, fasting_blood_sugar,
                    resting_ecg_results, maximum_heartrate, exercise_induced_angina, stdepression_induced,
                    slope, mayor_vessels, thal]

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


        age = request.form['Age']
        sex = request.form['Sex']
        chest_pain = request.form['Chest_Pain']
        resting_blood_pressure = request.form['Resting_Blood_Pressure']
        serum_cholestoral = request.form['Serum_Cholestoral']
        fasting_blood_sugar = request.form['Fasting_Blood_Sugar']
        resting_ecg_results = request.form['Resting_ECG_Results']
        maximum_heartrate = request.form['Maximum_HeartRate']
        exercise_induced_angina = request.form['Exercise_Induced_Angina']
        stdepression_induced = request.form['STDepression_Induced']
        slope = request.form['Slope']
        mayor_vessels = request.form['Mayor_Vessels']
        thal = request.form['Thal']
        req_model = request.form['req_model']

        target = get_predictions(age, sex, chest_pain, resting_blood_pressure, serum_cholestoral, fasting_blood_sugar,
                    resting_ecg_results, maximum_heartrate, exercise_induced_angina, stdepression_induced,
                    slope, mayor_vessels,thal, req_model)

        if target==1:
            status = 'Patient probably has a heart condition'
        else:
            status = 'Patient is completely fine'

        return render_template('home.html', target = target, status = status)
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(debug=True)