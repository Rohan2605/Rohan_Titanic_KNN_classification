import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import MinMaxScaler
import pickle


app = Flask(__name__)
model = pickle.load(open('KNN_classification_Titanic.pkl','rb'))
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
    
    sc = MinMaxScaler()
    prediction = model.predict(sc.transform([[Pclass, Sex, Age, Sibsp, Parch, Fare]]))
    
    if prediction == [0]:
      return render_template('index.html', prediction_text='Sorry, the person you are searching for is no more')
    
    else:
      return render_template('index.html', prediction_text='Thank GOD, the person you are searching for is safe ')

if __name__ == "__main__":
    app.run(debug = True)
