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
import os

PEOPLE_FOLDER = os.path.join('static', 'images')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER

@app.route('/')
def index():
  pic1 = os.path.join(app.config['UPLOAD_FOLDER'], 'pic1.jpg')
  pic2 = os.path.join(app.config['UPLOAD_FOLDER'], 'pic2.jpg')
  pic3 = os.path.join(app.config['UPLOAD_FOLDER'], 'pic3.jpg')
  return render_template("index.html", user_pic1 = pic1, user_pic2 = pic2, user_pic3 = pic3)

@app.route('/sol', methods=['POST','GET'])
def getDate():
  date=request.form['date']
  pic1 = os.path.join(app.config['UPLOAD_FOLDER'], 'pic1.jpg')
  pic2 = os.path.join(app.config['UPLOAD_FOLDER'], 'pic2.jpg')
  pic3 = os.path.join(app.config['UPLOAD_FOLDER'], 'pic3.jpg')
  solarOutput= solar_prediction.dataSetProcessing(date)
  return render_template("output.html",solarOutput=solarOutput,date=date, user_pic1 = pic1, user_pic2 = pic2, user_pic3 = pic3)

if __name__ == '_main_':
    app.run(debug=True)