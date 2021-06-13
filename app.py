from flask import Flask, render_template, request, send_from_directory
from flask_bs4 import Bootstrap
from forms import ClassificationForm, LearningForm
from classification import get_files, predict, allowed_file
from fitting import MLA
import os
from werkzeug.utils import secure_filename
import pandas as pd

UPLOAD_FOLDER = '/dataset/classification/'
ALLOWED_EXTENSIONS = {'csv'}
ALGORITHMS = ['DecisionTreeClassifier', 'DecisionTreeRegressor', 'KNeighborsClassifier',
'KNeighborsRegressor', 'SVMClassifier', 'SVMRegressor', 'NaiveBayes', 'LogisticRegression']
CRITERION_CLF = ['gini', 'entropy']
CRITERION_RGR = ['mae', 'friedman_mse', 'mse', 'poisson']
KERNEL = ['linear', 'rbf', 'sigmoid', 'precomputed']
PENALTY = ['l1', 'l2', 'elasticnet', 'none']
SOLVER = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
ALGORITHM_KNN = ['auto', 'ball_tree', 'kd_tree', 'brute']

app = Flask(__name__)
bootstrap = Bootstrap(app)

SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

global algo
algo = ""

@app.route("/")
def main_page():
    return render_template("main_page.html")

@app.route("/algorithms")
def algorithms():
    return render_template("algorithms.html")  

@app.route("/classification", methods=['GET', 'POST'])
def classification():
    form = ClassificationForm()
    models = get_files('./models', '.sav')
    form.classification_model.choices=models
    data = []
    predictions = []
    success = False
    error_found = False
    error = False
    error_csv = False
    check = False
    length = 0
    length_trgt = 0
    if form.validate_on_submit():
        input_file = request.files['input_file']
        if input_file and allowed_file(input_file.filename, ALLOWED_EXTENSIONS):
            filename = secure_filename(input_file.filename)
            input_file.save(os.path.join(filename))
        else:
            error_csv = True
            return render_template("classification.html",
            form=form,
            data=data,
            error_csv=error_csv)
        modelname = form.classification_model.data
        file = f'./dataset/{filename}' 
        length_trgt = len(pd.read_csv(file)["Label"].unique())
        if modelname.endswith('clf.sav') and length_trgt > 2:
            error = True
            return render_template("classification.html",
            form=form,
            data=data,
            error=error)
        elif modelname.endswith('rgr.sav') and length_trgt == 2:
            try:
                check = True
                data, predictions = predict(f'./models/{modelname}', file, check)
                length = len(predictions)
                success = True
            except:
                error_found = True
                error = False
                return 'Something goes wrong! Try again', 400
        else:
            try:
                check = False
                data, predictions = predict(f'./models/{modelname}', file, check)
                length = len(predictions)
                success = True
            except:
                error_found = True
                error = False
                return 'Something goes wrong! Try again', 400
    return render_template("classification.html",
        form=form, 
        data=data,
        predictions=predictions, 
        length=length,
        success=success,
        error_found=error_found,
        length_trgt=length_trgt,
        error=error,
        check=check,
        error_csv=error_csv) 

@app.route("/learning", methods=['GET', 'POST'])
def learning():
    form = LearningForm(prefix='form')
    success = False
    form.algorithms.choices = ALGORITHMS
    form.criterion_clf.choices = CRITERION_CLF
    form.criterion_rgr.choices = CRITERION_RGR
    form.algorithm_knn.choices = ALGORITHM_KNN
    form.kernel.choices = KERNEL
    form.penalty.choices = PENALTY
    form.solver.choices = SOLVER
    duration_of_fit = 0
    metric = 0
    is_clf = True
    criterion_clf = ""
    criterion_rgr = ""
    algorithm_knn = ""
    penalty = ""
    solver = ""
    kernel = ""
    max_depth = 0
    n_neighbors = 5
    c = 0
    max_iter = 0
    var_smoothing = 1e-15

    if form.choose_algo.data:
        global algo
        algo = form.algorithms.data      
    elif form.submit.data:
        if algo == 'DecisionTreeClassifier':
            max_depth = form.max_depth.data
            criterion_clf = form.criterion_clf.data
        elif algo == 'DecisionTreeRegressor':
            max_depth = form.max_depth.data
            criterion_rgr = form.criterion_rgr.data
        elif algo == 'KNeighborsClassifier' or algo == 'KNeighborsRegressor':
            n_neighbors = form.n_neighbors.data
            algorithm_knn = form.algorithm_knn.data
        elif algo == 'SVMClassifier' or algo == 'SVMRegressor':
            kernel = form.kernel.data
            c = form.c.data
            max_iter = form.max_iter.data
        elif algo == 'NaiveBayes':
            var_smoothing = form.var_smoothing.data
        elif algo == 'LogisticRegression':
            penalty = form.penalty.data
            solver = form.solver.data
        else:
            print ("Trouble with choice algorithm")
        label_name = form.label_name.data
        filename = secure_filename(form.input_file.data.filename)
        form.input_file.data.save(os.path.join(filename))
        file = f'./dataset/{filename}' 
        try:
            metric, duration_of_fit, is_clf = MLA(algo, file, label_name, duration_of_fit, max_depth, criterion_clf,
            criterion_rgr, int(n_neighbors), algorithm_knn, kernel, c, max_iter, var_smoothing, penalty, solver)
            success = True
        except Exception as e:
            return str(e), 400
    else:
        print (form.errors)

    return render_template(
        'learning.html', 
        form=form,
        success=success,
        metric=metric,
        duration_of_fit=duration_of_fit,
        is_clf=is_clf,
        algo=algo)  

@app.route("/dataset/classification/<path:filename>", methods=['GET', 'POST'])
def download(filename):
    directory = './dataset/classification/'
    return send_from_directory(directory, filename)


if __name__ == "__main__":
    app.run(debug=True)