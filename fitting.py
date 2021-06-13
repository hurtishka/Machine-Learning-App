import pandas as pd
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
import pickle
from timeit import default_timer as timer
from classification import label_encoder_data

def MLA(algo, file, label_name, duration_of_fit, max_depth, criterion_clf,
            criterion_rgr, n_neighbors, algorithm_knn, kernel, c, max_iter, var_smoothing, penalty, solver):
    data = pd.read_csv(file)
    encoded_data = label_encoder_data(data)
    encoded_data = encoded_data.sample(frac=1).reset_index(drop=True)
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
        model = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=algorithm_knn)
        start_fit = timer() 
        model.fit(X_train,y_train)
        end_fit = timer()
        duration_of_fit = end_fit - start_fit
        predictions = model.predict(X_valid)
        metric = accuracy_score(y_valid, predictions) 
    elif algo == 'KNeighborsRegressor':
        model = KNeighborsRegressor(n_neighbors=n_neighbors, algorithm=algorithm_knn)
        start_fit = timer() 
        model.fit(X_train,y_train)
        end_fit = timer()
        duration_of_fit = end_fit - start_fit
        predictions = model.predict(X_valid)
        metric = mean_absolute_error(y_valid, predictions) 
        is_clf = False
    elif algo == 'SVMClassifier':
        model = SVC(kernel=kernel, C=c, max_iter=max_iter)
        start_fit = timer() 
        model.fit(X_train,y_train)
        end_fit = timer()
        duration_of_fit = end_fit - start_fit
        predictions = model.predict(X_valid)
        metric = accuracy_score(y_valid, predictions) 
    elif algo == 'SVMRegressor':
        model = SVR(kernel=kernel, C=c, max_iter=max_iter)
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
        
    pickle.dump(model, open('./dataset/classification/model.sav', 'wb'))
    metric = "{0:.2f}".format(metric*100)
    duration_of_fit = "{0:.4f} секунд".format(duration_of_fit)

    return metric, duration_of_fit, is_clf