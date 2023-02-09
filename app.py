import urllib.request
import sys
import csv
import codecs
import pandas as pd
from flask import Flask, render_template
import pandas as pd
import numpy as np
from flask import render_template


app = Flask(__name__)

@app.route('/')
def index():
  return render_template('index.html')


if __name__ == '_main_':
    app.run(debug=True)