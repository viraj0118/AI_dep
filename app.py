import numpy as np
from flask import Flask, request, jsonify, render_template
#from flask_ngrok import run_with_ngrok
import pickle
import pandas as pd


app = Flask(__name__)


#run_with_ngrok(app)

@app.route('/')
def home():
  
    return render_template("index.html")
  
@app.route('/predict',methods=['GET'])
def predict():
    
    
    '''
    For rendering results on HTML GUI
    '''
    credit = float(request.args.get('credit'))
    geography=float(request.args.get('geography'))
    gender=float(request.args.get('gender'))
    age=float(request.args.get('age'))
    tenure=float(request.args.get('tenure'))
    balance=float(request.args.get('balance'))
    num_p=float(request.args.get('num_p'))
    card=float(request.args.get('card'))
    model1=float(request.args.get('model1'))
    active=float(request.args.get('active'))
    salary=float(request.args.get('salary'))

    if model1==0:
      model=pickle.load(open('Project7/project7_decision_model.pkl','rb'))
    elif model1==1:
      model=pickle.load(open('Project7/project7_svm.pkl','rb'))
    elif model1==2:
      model=pickle.load(open('Project7/project7_random_forest.pkl','rb'))
    elif model1==3:
      model=pickle.load(open('Project7/project7_knn.pkl','rb'))
    elif model1==4:
      model=pickle.load(open('Project7/project7_naive.pkl','rb'))
      

    dataset= pd.read_csv('Project7/Churn_Modelling.csv')
    X = dataset.iloc[:, 3:13].values
    from sklearn.preprocessing import LabelEncoder
    labelencoder_X = LabelEncoder()
    X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
    X[:, 2] = labelencoder_X.fit_transform(X[:, 2])

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X = sc.fit_transform(X)
    prediction = model.predict(sc.transform([[credit,geography,gender,age,tenure,balance,num_p,card,active,salary]]))
    if prediction==0:
      message="Not Exited"
    else:
      message="Exited"
    
        
    return render_template('index.html', prediction_text='Model  has predicted : {}'.format(message))


if __name__=='__main__':
    app.run(host='0.0.0.0',port=8000)
