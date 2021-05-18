from flask import Flask, render_template
from flask_bs4 import Bootstrap

app = Flask(__name__)
bootstrap = Bootstrap(app)

@app.route("/")
def main_page():
    return render_template("main_page.html")

@app.route("/algorithms")
@app.route("/algorithms/<algorithms_id>")
def algorithms(algorithms_id=0):
    return render_template("algorithms.html",algorithms_id=algorithms_id)  

@app.route("/classif_train")
def classification():
    
    return render_template("classification.html")  

if __name__ == "__main__":
    app.run(debug=True)