from flask import Flask
from flask import request, render_template
import pandas as pd
import joblib


app = Flask(__name__)

@app.route("/", methods = ["GET", "POST"])
def i():
    if request.method == "POST":
        
        age = None
        income = None
        loan = None

        age = request.form.get("age")
        income = request.form.get("income")
        loan = request.form.get("loan")

        print("age " + age)
        print("income " + income)
        print("loan "+ loan)

        age_mean = 34.795949606582
        age_sd = 12.840054954894491

        income_mean = 45136.87597485758
        income_sd = 14425.486619355603

        loan_mean = 5591.986694851936
        loan_sd = 3174.522430458773

        age_normalized = (float(age) - age_mean)/age_sd
        income_normalized = (float(income) - income_mean)/income_sd
        loan_normalized = (float(loan) - loan_mean)/loan_sd

        print(age_normalized)
        print(income_normalized)
        print(loan_normalized)

        regressionmodel = joblib.load("Regression")
        decisiontreemodel = joblib.load("decisiontree")
        randomforestmodel = joblib.load("randomforest")
        xgboostmodel = joblib.load("xgboost")
        neuralnetmodel = joblib.load("neuralnet")

    

        regressionpred = regressionmodel.predict([[float(age), float(loan), float(income)]])
        decisiontreepred = decisiontreemodel.predict([[float(age), float(loan), float(income)]])
        randomforestpred = randomforestmodel.predict([[float(age), float(loan), float(income)]])
        xgboostpred = xgboostmodel.predict([[float(age), float(loan), float(income)]])
        nnpred = neuralnetmodel.predict([[age_normalized, loan_normalized, income_normalized]])

        print(regressionpred)
        print(decisiontreepred)
        print(randomforestpred)
        print(xgboostpred)
        print(nnpred)

        return(render_template("index.html", result1 = regressionpred, result2=decisiontreepred, result3=randomforestpred, result4=xgboostpred, result5=nnpred))
    else:
        return(render_template("index.html"))

if __name__=="__main__":
    app.run()

