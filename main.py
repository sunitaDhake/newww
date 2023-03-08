# new line added at title of file
from flask import Flask,render_template,request,jsonify,redirect,url_for
import numpy as np
import json
import pickle

with open('artifacts\project_data.json','r') as file:
    project_data = json.load(file)



with open('artifacts\scale.pkl','rb') as file:
    scaler = pickle.load(file)


with open('artifacts\model.pkl','rb') as file:
    model = pickle.load(file)


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_data', methods = ['POST'])
def get_data():
    data = request.form
    # result = Charges_prediction(data)
    age = data['html_age']
    gender = data['html_gender']
    bmi = data['html_bmi']
    smoker = data['html_smoker']
    region = data['html_region']
           
    user_data = np.zeros(len(project_data['column_names']))
    user_data[0] = age
    user_data[1] = project_data['gender'][gender]
    user_data[2] = bmi
    user_data[3] = project_data['smoker'][smoker]

    search_region = 'region_'+region
    index = np.where(np.array(project_data['column_names']) == search_region)[0][0]
    user_data[index] = 1

    ### Scaling the user data 
    user_data_scale = scaler.transform([user_data])
    print(user_data_scale)

    result = model.predict(user_data_scale)[0]
    print(result)
    return render_template('index.html',prediction =result)

if __name__ == "__main__":
    app.run(host = '0.0.0.0', port=5000, debug =True)