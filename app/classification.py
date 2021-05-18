import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def get_data(data, target):
    df = pd.read_csv(data)
    X = df.drop([target], axis=1)
    y = df[target].astype("int")
    return X,y

def train(X, y, test_size, algorithm):
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size, random_state=1)
    if algorithm == "DecisionTree":
        model = DecisionTreeClassifier()
        model.fit(X_train,y_train)
    elif algorithm == "KNearestNeighbors":
        model = KNeighborsClassifier()
        model.fit(X_train,y_train)
    elif algorithm == "SVM":
        model = SVC()
        model.fit(X_train,y_train)
    elif algorithm == "NaiveBayes":
        model = GaussianNB()
        model.fit(X_train,y_train)
    elif algorithm == "LogisticRegression":
        model = LogisticRegression()
        model.fit(X_train,y_train) 
    else:
        print("Error")
    return model, X_valid, y_valid

def predict(model, X_valid, y_valid):
    prediction_model = model.predict(X_valid)
    report = classification_report(y_valid, prediction_model)
    return prediction_model, report
