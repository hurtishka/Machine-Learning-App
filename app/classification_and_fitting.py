import pandas as pd
from os import listdir
from pandas import read_csv
from pickle import load
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
from timeit import default_timer as timer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression


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
        data = read_csv(file)
        encoded_data = label_encoder_data(data)
        df = encoded_data.drop("Label", axis=True)
        df = shuffle(df)
        predictions = model.predict(df)
        predictions = np.around(predictions, decimals=0)
        predictions = np.where(predictions == 2., 1., predictions)
    else:    
        model = load_model(model_path)
        data = read_csv(file)
        encoded_data = label_encoder_data(data)
        df = encoded_data.drop("Label", axis=True)
        df = shuffle(df)
        predictions = model.predict(df)
        predictions = np.around(predictions, decimals=0)
    return data, predictions

def get_files(folder_name, file_extension):
    files = listdir(folder_name)
    files = list(filter(lambda x: x.endswith(file_extension), files))
    return files
    
def allowed_file(filename, ALLOWED_EXTENSIONS):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def MLA(algo, file, label_name, duration_of_fit, max_depth, criterion_clf,
            criterion_rgr, n_neighbors, algorithm_knn, kernel, c_svm, max_iter, var_smoothing, penalty, solver):
    data = read_csv(file)
    encoded_data = label_encoder_data(data)
    encoded_data = shuffle(encoded_data)
    y = encoded_data[label_name]
    X = encoded_data.drop(label_name, axis=True)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=1)
    is_clf = True
    if algo == 'DecisionTreeClassifier':
        model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion_clf)
        start_fit = timer() 
        model.fit(X_train,y_train)
        end_fit = timer()
        duration_of_fit = end_fit - start_fit
        predictions = model.predict(X_valid)
        metric = accuracy_score(y_valid, predictions)
    elif algo == 'DecisionTreeRegressor':
        model = DecisionTreeRegressor(max_depth=max_depth, criterion=criterion_rgr)
        start_fit = timer() 
        model.fit(X_train,y_train)
        end_fit = timer()
        duration_of_fit = end_fit - start_fit
        predictions = model.predict(X_valid)
        metric = mean_absolute_error(y_valid, predictions) 
        is_clf = False
    elif algo == 'KNeighborsClassifier':
        model = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm_knn=algorithm_knn)
        start_fit = timer() 
        model.fit(X_train,y_train)
        end_fit = timer()
        duration_of_fit = end_fit - start_fit
        predictions = model.predict(X_valid)
        metric = accuracy_score(y_valid, predictions) 
    elif algo == 'KNeighborsRegressor':
        model = KNeighborsRegressor(n_neighbors=n_neighbors, algorithm_knn=algorithm_knn)
        start_fit = timer() 
        model.fit(X_train,y_train)
        end_fit = timer()
        duration_of_fit = end_fit - start_fit
        predictions = model.predict(X_valid)
        metric = mean_absolute_error(y_valid, predictions) 
        is_clf = False
    elif algo == 'SVMClassifier':
        model = SVC(kernel=kernel, c_svm=c_svm, max_iter=max_iter)
        start_fit = timer() 
        model.fit(X_train,y_train)
        end_fit = timer()
        duration_of_fit = end_fit - start_fit
        predictions = model.predict(X_valid)
        metric = accuracy_score(y_valid, predictions) 
    elif algo == 'SVMRegressor':
        model = SVR(kernel=kernel, c_svm=c_svm, max_iter=max_iter)
        start_fit = timer() 
        model.fit(X_train,y_train)
        end_fit = timer()
        duration_of_fit = end_fit - start_fit
        predictions = model.predict(X_valid)
        metric = mean_absolute_error(y_valid, predictions) 
        is_clf = False
    elif algo == 'NaiveBayes':
        model = GaussianNB(var_smoothing=var_smoothing)
        start_fit = timer() 
        model.fit(X_train,y_train)
        end_fit = timer()
        duration_of_fit = end_fit - start_fit
        predictions = model.predict(X_valid)
        metric = accuracy_score(y_valid, predictions) 
    elif algo == 'LogisticRegression':
        model = LogisticRegression(penalty=penalty, solver=solver)
        start_fit = timer() 
        model.fit(X_train,y_train)
        end_fit = timer()
        duration_of_fit = end_fit - start_fit
        predictions = model.predict(X_valid)
        metric = accuracy_score(y_valid, predictions) 
    else:
        print("Error while fitting!")
    metric = "{0:.2f}".format(metric*100)
    duration_of_fit = "{0:.4f} секунд".format(duration_of_fit)
    print(metric)

    return metric, duration_of_fit, is_clf