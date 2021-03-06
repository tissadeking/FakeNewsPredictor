#Importing the Libraries
import numpy as np
from flask import Flask, request,render_template
from flask_cors import CORS
import os
import pickle
import flask
import os
import newspaper
from newspaper import Article
import urllib


app = Flask(__name__)
CORS(app)
app=flask.Flask(__name__, template_folder='templates')

with open('model.pickle', 'rb') as handle:
	model = pickle.load(handle)

@app.route('/')
def main():
    return render_template('index.html')
@app.route('/predict',methods = ['GET', 'POST'])
def predict():
    url = request.get_data(as_text = True)[5:]
    url = urllib.parse.unquote(url)
    article = Article(str(url))
    article.download()
    article.parse()
    article.nlp()
    news = article.summary

    label_prediction = model.predict([news])
    return render_template('index.html', prediction_text = 'Result: {}'.format(label_prediction[0]))

if __name__ == "__main__":
    port = int(os.environ.get('port', 5000))
    app.run(port=port, debug=True, use_reloader=False)
