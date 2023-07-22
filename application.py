from flask import Flask,request,render_template,jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline



application=Flask(__name__)

app=application

## Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])

def predict_datapoint():
        if request.method=='GET':
            return render_template('form.html')
        else:
             data=CustomData(
            src_bytes=float(request.form.get('src_bytes')),
            dst_host_serror_rate = float(request.form.get('dst_host_serror_rate')),
            count = float(request.form.get('count')),
            dst_bytes = float(request.form.get('dst_bytes')),
            same_srv_rate = float(request.form.get('same_srv_rate')),
            dst_host_same_srv_rate = float(request.form.get('dst_host_same_srv_rate')),
            logged_in = float(request.form.get('logged_in'))
        )
        final_new_data=data.get_data_as_dataframe()
        print(final_new_data)

        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(final_new_data)

        return render_template('form.html',final_result=results)
    

if __name__=="__main__":
    app.run(host='0.0.0.0')