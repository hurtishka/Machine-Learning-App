import pandas as pd
from os import listdir
from pandas import read_csv
from pickle import load
from sklearn.preprocessing import LabelEncoder



def label_encoding(data):
    columns_to_encode = list(data.select_dtypes(include=['category', 'object']))
    le = LabelEncoder()
    for feature in columns_to_encode:
        try:
            data[feature] = le.fit_transform(data[feature])
        except:
            print('error' + feature)
    return data


def load_model(path):
    with open(path, "rb") as f:
        try:
            loaded_model = load(f)
        except:
            print('Somethind wrong with the input file.')
    return loaded_model


def predict(model_path, file):
    model = load_model(model_path)
    print("7")
    data = read_csv(file)
    print("8")
    #enconded_data = label_encoding(data)
    print("9")
    df = data.drop("Label", axis=True)
    predictions = model.predict(df)
    print("10")
    return data, predictions

def get_files_from_root(folder_name, file_extension):
    files = listdir(folder_name)
    files = list(filter(lambda x: x.endswith(file_extension), files))
    return files


def allowed_file(filename, ALLOWED_EXTENSIONS):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS