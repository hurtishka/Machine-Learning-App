import pandas as pd
from os import listdir
from pandas import read_csv
from pickle import load
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import numpy as np

def label_encoding(data):
    columns_to_encode = list(data.select_dtypes(include=['object']))
    le = LabelEncoder()
    for feature in columns_to_encode:
        data[feature] = le.fit_transform(data[feature])
    return data


def load_model(path):
    with open(path, "rb") as f:
        loaded_model = load(f)
    return loaded_model


def predict(model_path, file, check):
    if check == True:
        model = load_model(model_path)
        data = read_csv(file)
        encoded_data = label_encoding(data)
        df = encoded_data.drop("Label", axis=True)
        predictions = model.predict(df)
        predictions = np.around(predictions, decimals=0)
        predictions = np.where(predictions == 2., 1., predictions)
    else:    
        model = load_model(model_path)
        data = read_csv(file)
        encoded_data = label_encoding(data)
        df = encoded_data.drop("Label", axis=True)
        predictions = model.predict(df)
        predictions = np.around(predictions, decimals=0)
    return data, predictions

def get_files(folder_name, file_extension):
    files = listdir(folder_name)
    files = list(filter(lambda x: x.endswith(file_extension), files))
    return files


def allowed_file(filename, ALLOWED_EXTENSIONS):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS