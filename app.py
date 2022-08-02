import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import StandardScaler
import pickle


app = Flask(__name__)
model = pickle.load(open('KNN_classification_Titanic.pkl','rb'))
dataset= pd.read_csv('train.csv')
@app.route('/')
def home():
  
    return render_template("index.html")
  
@app.route('/predict',methods=['GET'])
def predict():
    
    age = int(request.args.get('age'))
    fare = float(request.args.get('fare'))
    parch = int(request.args.get('parch'))
    pclass = int(request.args.get('pclass'))
    sibsp = int(request.args.get('sibsp'))
    sex = (request.args.get('sex'))

    if sex=="Male":
      sex = 1
    else:
      sex = 0
      
    x = dataset.iloc[:, [2,4,5,6,7,9]].values
    
    sc = StandardScaler()
    x = sc.fit_transform(x)
    prediction = model.predict(sc.transform([[pclass, sex, age, sibsp, parch, fare]]))
    
    if prediction == [0]:
      return render_template('index.html', prediction_text='Sorry, the person you are searching for is no more')
    
    else:
      return render_template('index.html', prediction_text='Thank GOD, the person you are searching for is safe ')

if __name__ == "__main__":
    app.run(debug = True)
