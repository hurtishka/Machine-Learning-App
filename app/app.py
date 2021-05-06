from flask import Flask, render_template
import classification
app = Flask(__name__)
@app.route("/")
def main_page():
    return render_template("main_page.html")

@app.route("/algorithms")
@app.route("/algorithms/<algorithms_id>")
def algorithms(algorithms_id=0):
    return render_template("algorithms.html",algorithms_id=algorithms_id)  

@app.route("/classification")
def classification():
    
    return render_template("classification.html")  

if __name__ == "__main__":
    app.run(debug=True)