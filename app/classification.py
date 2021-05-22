import pandas as pd

def get_data(data, target):
    df = pd.read_csv(data)
    X = df.drop([target], axis=1)
    return X

def predict(model, X):
    prediction_model = model.predict(X)
    return prediction_model
