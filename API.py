import urllib.request
import sys

import csv
import codecs
import pandas as pd
        
try: 
  ResultBytes = urllib.request.urlopen("https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/17.4116%2C%2078.3987/next7days?unitGroup=metric&include=hours&key=HTXNURB3NT3SGJHKHUBK63QGK&contentType=csv")
  
  # Parse the results as CSV
  CSVText = pd.read_csv(ResultBytes)
  CSVText.to_csv('predicted_weather.csv')
        
except urllib.error.HTTPError  as e:
  ErrorInfo= e.read().decode() 
  print('Error code: ', e.code, ErrorInfo)
  sys.exit()
except  urllib.error.URLError as e:
  ErrorInfo= e.read().decode() 
  print('Error code: ', e.code,ErrorInfo)
sys.exit()