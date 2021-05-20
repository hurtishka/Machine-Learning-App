from flask import Flask, render_template
from flask_bs4 import Bootstrap

app = Flask(__name__)
bootstrap = Bootstrap(app)

@app.route("/")
def main_page():
    return render_template("main_page.html")

@app.route("/algorithms")
def algorithms(algorithms_id=0):
    return render_template("algorithms.html", algorithms_id=algorithms_id)  

@app.route("/classification_train")
def classification():
    
    return render_template("classification_train.html")  

if __name__ == "__main__":
    app.run(debug=True)