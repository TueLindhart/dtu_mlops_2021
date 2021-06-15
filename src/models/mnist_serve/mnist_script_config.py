import json
import joblib
import numpy as np
import torch
from azureml.core.model import Model

# Called when the service is loaded


def init():
    global model
    # Get the path to the deployed model file and load it
    model_path = Model.get_model_path('mnist_model')
    model = joblib.load(model_path)

# Called when a request is received


def run(raw_data):

    # Get the input data as a numpy array
    data = torch.Tensor(json.loads(raw_data)['data'])

    # Get a prediction from the model
    predictions = model(data)

    return json.dumps(predictions)
