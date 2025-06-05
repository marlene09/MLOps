import json
import numpy as np
from azureml.core.model import Model
import joblib

import pickle
import os

# Local path to the model file
# def init():
#     global model
#     model_path = os.path.join(os.getcwd(), '/Users/marlenepostop/MLOps/diabetes-prod-model_/model.pkl')  # directly reference model.pkl
#     with open(model_path, 'rb') as f:
#         model = pickle.load(f)

def init():
    global model
    model_path = Model.get_model_path("diabetes-prod-model")  # Must match Azure model name
    model = joblib.load(model_path)
    
def run(raw_data):
    try:
        data = json.loads(raw_data)
        # Convert data to 2D array if it's a single sample of features
        # For example, if data is a dict or list of feature values
        if isinstance(data, dict):
            # extract values and wrap in list
            data = [list(data.values())]
        elif isinstance(data, list):
            # if a list of feature values, wrap to 2D
            if len(data) > 0 and not isinstance(data[0], list):
                data = [data]

        data = np.array(data)

        prediction = model.predict(data)
        return prediction.tolist()

    except Exception as e:
        return str(e)
