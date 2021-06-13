import pandas as pd
from os import listdir
from pandas import read_csv
from pickle import load
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd



def label_encoder_data(data):
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
        data = pd.read_csv(file)
        encoded_data = label_encoder_data(data)
        df = encoded_data.drop("Label", axis=True)
        df = df.sample(frac=1).reset_index(drop=True)
        predictions = model.predict(df)
        predictions = np.around(predictions, decimals=0)
        predictions = np.where(predictions == 2., 1., predictions)
        new_data = df.copy()
        try:
            new_data['Predict'] = predictions
            new_data.to_csv('dataset/classification/results.csv', index=False)
        except:
            print('Bad request')
    else:    
        model = load_model(model_path)
        data = pd.read_csv(file)
        encoded_data = label_encoder_data(data)
        df = encoded_data.drop("Label", axis=True)
        df = df.sample(frac=1).reset_index(drop=True)
        predictions = model.predict(df)
        predictions = np.around(predictions, decimals=0)
        new_data = df.copy()
        try:
            new_data['Predict'] = predictions
            new_data.to_csv('dataset/classification/results.csv', index=False)
        except:
            print('Bad request')
    return df, predictions

def get_files(folder_name, file_extension):
    files = listdir(folder_name)
    files = list(filter(lambda x: x.endswith(file_extension), files))
    return files
    
def allowed_file(filename, ALLOWED_EXTENSIONS):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

