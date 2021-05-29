from flask import Flask, render_template, request
from flask_bs4 import Bootstrap
from forms import ClassificationForm
from classification import get_files, allowed_file, predict
import os
from werkzeug.utils import secure_filename
import pandas as pd
UPLOAD_FOLDER = './dataset/'
ALLOWED_EXTENSIONS = set(['csv'])

app = Flask(__name__)
bootstrap = Bootstrap(app)

SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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
    check = False
    length = 0
    length_trgt = 0
    if form.validate_on_submit():
        input_file = request.files['input_file']
        if input_file and allowed_file(input_file.filename, ALLOWED_EXTENSIONS):
            filename = secure_filename(input_file.filename)
            input_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
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
        check=check) 

@app.route("/train", methods=['GET', 'POST'])
def train():
    return render_template("train.html")  

if __name__ == "__main__":
    app.run(debug=True)