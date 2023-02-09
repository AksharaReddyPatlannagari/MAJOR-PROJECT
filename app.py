import urllib.request
import sys
import csv
import codecs
import pandas as pd
from flask import Flask, render_template,request
import pandas as pd
import numpy as np
from flask import render_template
import cgi
from datetime import datetime
import solar_prediction 

app = Flask(__name__)

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/sol', methods=['POST','GET'])
def getDate():
  date=request.form['date']
  return solar_prediction.dataSetProcessing(date)

if __name__ == '_main_':
    app.run(debug=True)