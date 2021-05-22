from flask import Flask, render_template
from flask_bs4 import Bootstrap

app = Flask(__name__)
bootstrap = Bootstrap(app)

@app.route("/")
def main_page():
    return render_template("main_page.html")

@app.route("/algorithms")
def algorithms():
    return render_template("algorithms.html")  

@app.route("/classification", methods=['GET', 'POST'])
def classification():
    return render_template("classification.html") 

@app.route("/train", methods=['GET', 'POST'])
def train():
    return render_template("train.html")  

if __name__ == "__main__":
    app.run(debug=True)