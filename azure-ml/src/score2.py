import json
import numpy as np
import os
import pickle
import joblib

def init():
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It's the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION).
    # For multiple models, it points to the folder containing all deployed models (./azureml-models).
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'digits_model.pkl')
    model = joblib.load(model_path)

def run(raw_data):
    data = np.array(json.loads(raw_data)['data'])
    # Make prediction.
    y_hat = model.predict(data)
    # setosa_clases = ['Setosa', 'Versicolor', 'Virginica']
    # You can return any data type as long as it's JSON-serializable.
    # return json.dumps({'predicted class':setosa_clases[int(y_hat)]})
    return json.dumps({'predicted class': int(y_hat)})